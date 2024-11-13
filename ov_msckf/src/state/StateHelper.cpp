/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "StateHelper.h"

#include "state/State.h"

#include "types/Landmark.h"
#include "utils/colors.h"
#include "utils/print.h"
#include <cmath>
#include <numeric>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <thread>
#include <future>
using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;


// 声明全局变量
//Eigen::VectorXd dx_global;
//std::mutex state_mutex;
// 故障诊断静态成员变量初始化


void StateHelper::EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                                 const std::vector<std::shared_ptr<Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                                 const Eigen::MatrixXd &Q) {

  // We need at least one old and new variable
  if (order_NEW.empty() || order_OLD.empty()) {
    PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Loop through our Phi order and ensure that they are continuous in memory
  int size_order_NEW = order_NEW.at(0)->size();
  for (size_t i = 0; i < order_NEW.size() - 1; i++) {
    if (order_NEW.at(i)->id() + order_NEW.at(i)->size() != order_NEW.at(i + 1)->id()) {
      PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with non-contiguous state elements!\n" RESET);
      PRINT_ERROR(
          RED "StateHelper::EKFPropagation() - This code only support a state transition which is in the same order as the state\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    size_order_NEW += order_NEW.at(i + 1)->size();
  }

  // Size of the old phi matrix
  int size_order_OLD = order_OLD.at(0)->size();
  for (size_t i = 0; i < order_OLD.size() - 1; i++) {
    size_order_OLD += order_OLD.at(i + 1)->size();
  }

  // Assert that we have correct sizes
  assert(size_order_NEW == Phi.rows());
  assert(size_order_OLD == Phi.cols());
  assert(size_order_NEW == Q.cols());
  assert(size_order_NEW == Q.rows());

  // Get the location in small phi for each measuring variable
  int current_it = 0;
  std::vector<int> Phi_id;
  for (const auto &var : order_OLD) {
    Phi_id.push_back(current_it);
    current_it += var->size();
  }

  // Loop through all our old states and get the state transition times it
  // Cov_PhiT = [ Pxx ] [ Phi' ]'
  Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(), Phi.rows());
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Cov_PhiT.noalias() +=
        state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size()) * Phi.block(0, Phi_id[i], Phi.rows(), var->size()).transpose();
  }

  // Get Phi_NEW*Covariance*Phi_NEW^t + Q
  Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Phi_Cov_PhiT.noalias() += Phi.block(0, Phi_id[i], Phi.rows(), var->size()) * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
  }

  // We are good to go!
  int start_id = order_NEW.at(0)->id();
  int phi_size = Phi.rows();
  int total_size = state->_Cov.rows();
  state->_Cov.block(start_id, 0, phi_size, total_size) = Cov_PhiT.transpose();
  state->_Cov.block(0, start_id, total_size, phi_size) = Cov_PhiT;
  state->_Cov.block(start_id, start_id, phi_size, phi_size) = Phi_Cov_PhiT;

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }
}

void StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  
  std::cout<<"EKFUpdate:start:"<< state->_timestamp << std::endl;
 
  assert(res.rows() == R.rows());
  assert(H.rows() == res.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order[i];
      M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                       H.block(0, H_id[i], H.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }
//==========================================================
  //==========================================================
  // Get covariance of the involved terms
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // Residual covariance S = H*Cov*H' + R
  Eigen::MatrixXd S(R.rows(), R.rows());
  S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
  S.triangularView<Eigen::Upper>() += R;
  // Eigen::MatrixXd S = H * P_small * H.transpose() + R;

  // Invert our S (should we use a more stable method here??)
  Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
  S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
  Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();
  // Eigen::MatrixXd K = M_a * S.inverse();

  // Update Covariance
  //std::lock_guard<std::mutex> lock(state_mutex);
  state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  // Cov -= K * M_a.transpose();
  // Cov = 0.5*(Cov+Cov.transpose());

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }

  // Calculate our delta and update all our active states
  Eigen::VectorXd dx = K * res;
  // dx_global = dx;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
  }


  // If we are doing online intrinsic calibration we should update our camera objects
  // NOTE: is this the best place to put this update logic??? probably..
  if (state->_options.do_calib_camera_intrinsics) {
    for (auto const &calib : state->_cam_intrinsics) {
      state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
    }
  }
}
void StateHelper::GpsUpdate(std::shared_ptr<State> state, const Eigen::MatrixXd &H, const Eigen::VectorXd &res, const Eigen::MatrixXd &R,Eigen::MatrixXd &P_updated, Eigen::VectorXd &x_updated)
{
 // std::cout<<"GpsUpdate:start:"<< state->_timestamp << std::endl;

  const Eigen::MatrixXd P_minus = StateHelper::get_full_covariance(state);
  const Eigen::MatrixXd H_trans = H.transpose();
  const Eigen::MatrixXd S = H * P_minus * H_trans+R ;
  const Eigen::MatrixXd S_inv = S.llt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  const Eigen::MatrixXd K = P_minus * H_trans * S_inv;
  const Eigen::VectorXd delta_x = K * res;
  
 // std::cout<<"delta_x::"<< delta_x.block(state->_imu->id(), 0, state->_imu->size(), 1).transpose() << std::endl;

  // Update covariance.
  const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(state->_Cov.rows(), state->_Cov.rows()) - K * H;
  P_updated = I_KH * P_minus * I_KH.transpose() + K * R * K.transpose();
   P_updated =  P_updated.eval().selfadjointView<Eigen::Upper>();
   LimitMinDiagValue(1e-12, &P_updated);
  x_updated=delta_x;
/*   state->_Cov = I_KH * P_minus* I_KH.transpose() + K * R * K.transpose();
 
   //矩阵是对称的，并且只保留其上三角部分
   state->_Cov = state->_Cov.eval().selfadjointView<Eigen::Upper>();
   LimitMinDiagValue(1e-12, &state->_Cov);  */

/* for (size_t i = 0; i < state->_variables.size(); i++) {
    state->_variables.at(i)->update(delta_x.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
 } */
  //state->_imu->update(delta_x.block(state->_imu->id(), 0, state->_imu->size(), 1));
  

}
void StateHelper::ImmFilter(std::shared_ptr<State> state, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

    auto rT0_1 = boost::posix_time::microsec_clock::local_time();

    // 获取前一时刻的模型概率，扩展为三个模型
    double mp[3];
    mp[0] = state->p_huber1;    // 对应测量噪声为 2R 的模型
    mp[1] = state->p_huber2;     // 对应测量噪声为 R 的模型
    mp[2] = state->p_huber3;  // 对应测量噪声为 0.5R 的模型

    // 定义模型转移概率矩阵，大小为 3x3
    double trans_prob[3][3] = {
        {0.8, 0.1, 0.1},
        {0.1, 0.8, 0.1},
        {0.1, 0.1, 0.8}
    };

    // 计算预测的模型概率
    double pred_probs[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            pred_probs[j] += trans_prob[i][j] * mp[i];
        }
    }

    // 计算混合概率
/*     double mix_prob[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mix_prob[i][j] = (trans_prob[i][j] * mp[i]) / pred_probs[j];
        }
    } */

    // 混合初始化（由于预测步骤相同，可以跳过混合步骤）
    // 我们假设预测步骤已由 VIO 更新完成，直接使用当前状态作为先验
    //std::shared_ptr<State> state_huber = std::make_shared<State>(*state);
//propagator->propagate_and_clone_gps(state_huber, state->_timestamp-0.01);
    // 创建三个状态副本用于滤波
    std::shared_ptr<State> state_huber_2R = std::make_shared<State>(*state);
    std::shared_ptr<State> state_huber_R = std::make_shared<State>(*state);
    std::shared_ptr<State> state_huber_0_5R = std::make_shared<State>(*state);

    // 调整测量噪声协方差矩阵
    Eigen::MatrixXd R_2R = 1.75 * R;
    Eigen::MatrixXd R_R = R;
    Eigen::MatrixXd R_0_5R = 0.5 * R;

    // 执行滤波器更新
    Eigen::MatrixXd P_huber_2R, P_huber_R, P_huber_0_5R;
    Eigen::VectorXd x_huber_2R, x_huber_R, x_huber_0_5R;
/*  std::cout << "HUBER2r 1: " << state_huber_2R->_imu->pos() << std::endl;
 std::cout << "imm 1: " << state->_imu->pos() << std::endl; */
    // 创建线程并行运行三个滤波器
    std::future<void> huber_2R_future = std::async(std::launch::async, 
        HuberFilter, state_huber_2R, H, res, R_2R, std::ref(P_huber_2R), std::ref(x_huber_2R));
    std::future<void> huber_R_future = std::async(std::launch::async, 
        HuberFilter, state_huber_R, H, res, R_R, std::ref(P_huber_R), std::ref(x_huber_R));
    std::future<void> huber_0_5R_future = std::async(std::launch::async, 
        HuberFilter, state_huber_0_5R, H, res, R_0_5R, std::ref(P_huber_0_5R), std::ref(x_huber_0_5R));
 /* std::cout << "HUBER2r 2: " << state_huber_2R->_imu->pos() << std::endl;
  std::cout << "imm 2: " << state->_imu->pos() << std::endl; */
    huber_2R_future.get();
    huber_R_future.get();
    huber_0_5R_future.get();
    auto rT0_2 = boost::posix_time::microsec_clock::local_time();

    // 计算似然度
    Eigen::MatrixXd S_huber_2R = H * P_huber_2R * H.transpose() + R_2R;
    Eigen::MatrixXd S_huber_R = H * P_huber_R * H.transpose() + R_R;
    Eigen::MatrixXd S_huber_0_5R = H * P_huber_0_5R * H.transpose() + R_0_5R;

    double det_S_huber_2R = S_huber_2R.determinant();
    double det_S_huber_R = S_huber_R.determinant();
    double det_S_huber_0_5R = S_huber_0_5R.determinant();

    double likelihood_huber_2R = (1.0 / std::sqrt(std::pow(2 * M_PI, res.size()) * det_S_huber_2R)) *
                                 std::exp(-0.5 * res.transpose() * S_huber_2R.inverse() * res);

    double likelihood_huber_R = (1.0 / std::sqrt(std::pow(2 * M_PI, res.size()) * det_S_huber_R)) *
                                std::exp(-0.5 * res.transpose() * S_huber_R.inverse() * res);

    double likelihood_huber_0_5R = (1.0 / std::sqrt(std::pow(2 * M_PI, res.size()) * det_S_huber_0_5R)) *
                                   std::exp(-0.5 * res.transpose() * S_huber_0_5R.inverse() * res);

    // 更新模型概率
    double total_likelihood = pred_probs[0] * likelihood_huber_2R +
                              pred_probs[1] * likelihood_huber_R +
                              pred_probs[2] * likelihood_huber_0_5R;

    mp[0] = (pred_probs[0] * likelihood_huber_2R) / total_likelihood;
    mp[1] = (pred_probs[1] * likelihood_huber_R) / total_likelihood;
    mp[2] = (pred_probs[2] * likelihood_huber_0_5R) / total_likelihood;

    // 保存更新后的模型概率
    state->p_huber1 = mp[0];
    state->p_huber2 = mp[1];
    state->p_huber3 = mp[2];

    // 记录模型概率到文件
    std::ofstream outfile;
    outfile.open("/home/zhou/workspace/data/result/imm_probabilities.txt", std::ios_base::app);
    outfile << std::setprecision(15) << state->_timestamp << " "
            << mp[0] << " " << mp[1] << " " << mp[2] << std::endl;
    outfile.close();

    // 合并状态
    Eigen::VectorXd x_combined = mp[0] * x_huber_2R + mp[1] * x_huber_R + mp[2] * x_huber_0_5R;
Eigen::Vector3d huber_position1 = state->_imu->pos()+x_huber_2R.segment<3>(3);
Eigen::Vector3d huber_position2 = state_huber_R->_imu->pos()+x_huber_R.segment<3>(3);
Eigen::Vector3d huber_position3 = state_huber_0_5R->_imu->pos()+x_huber_0_5R.segment<3>(3);
    // 更新状态变量
    for (size_t i = 0; i < state->_variables.size(); i++) {
        state->_variables.at(i)->update(x_combined.segment(state->_variables.at(i)->id(), state->_variables.at(i)->size()));
    }
/*  std::cout << "HUBER2r 3: " << state_huber_2R->_imu->pos() << std::endl;
 std::cout << "imm 3: " << state->_imu->pos() << std::endl; */
  // 保存 IMU 位置信息
Eigen::Vector3d imu_position = state->_imu->pos();


std::ofstream imm_pos_file("/home/zhou/workspace/data/result/imm_position.txt", std::ios_base::app);
std::ofstream huber_file("/home/zhou/workspace/data/result/huber2r.txt", std::ios_base::app);
std::ofstream huber_file2("/home/zhou/workspace/data/result/huberr.txt", std::ios_base::app);
std::ofstream huber_file3("/home/zhou/workspace/data/result/huber05r.txt", std::ios_base::app);

if (imm_pos_file.is_open() && huber_file.is_open() && huber_file2.is_open() && huber_file3.is_open()) {
    imm_pos_file << std::setprecision(15)
                 << state->_timestamp << " "
                 << imu_position.transpose() << std::endl;

    huber_file << std::setprecision(15)
               << state->_timestamp << " "
               << huber_position1.transpose() << std::endl;

    huber_file2 << std::setprecision(15)
                << state->_timestamp << " "
                << huber_position2.transpose() << std::endl;

    huber_file3 << std::setprecision(15)
                << state->_timestamp << " "
                << huber_position3.transpose() << std::endl;
} else {
    std::cerr << "无法打开一个或多个文件进行写入操作！" << std::endl;
} 
 imm_pos_file.close();
huber_file.close();
huber_file2.close();
huber_file3.close(); 


    // 合并协方差
    // P = Σ(wi * (Pi + (xi - x̄)(xi - x̄)ᵀ))
    Eigen::VectorXd delta_huber_2R = x_huber_2R - x_combined;
    Eigen::VectorXd delta_huber_R = x_huber_R - x_combined;
    Eigen::VectorXd delta_huber_0_5R = x_huber_0_5R - x_combined;

    Eigen::MatrixXd P_combined = mp[0] * (P_huber_2R + delta_huber_2R * delta_huber_2R.transpose()) +
                                 mp[1] * (P_huber_R + delta_huber_R * delta_huber_R.transpose()) +
                                 mp[2] * (P_huber_0_5R + delta_huber_0_5R * delta_huber_0_5R.transpose());

    // 更新协方差
    state->_Cov = P_combined;

    // 确保协方差矩阵对称
    state->_Cov = 0.5 * (state->_Cov + state->_Cov.transpose());

    // 限制协方差矩阵的最小对角线值
    StateHelper::LimitMinDiagValue(1e-8, &state->_Cov);

    auto rT0_3 = boost::posix_time::microsec_clock::local_time();
    double time_total1 = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
    double time_total2= (rT0_3 - rT0_1).total_microseconds() * 1e-6;
    std::cout << "HUBER time: " << time_total1 << std::endl;
    std::cout << "IMMFilter time: " << time_total2 << std::endl;
}



/**
 * @brief Applies the Huber filter to update the state and covariance matrix.
 *
 * This function implements the Huber filter, which is a robust statistical method used to reduce the influence of outliers in the state estimation process. The filter iteratively updates the state and covariance matrix based on the provided measurement model and residuals.
 *
 * @param state A shared pointer to the State object that contains the current state and covariance matrix.
 * @param H The measurement matrix.
 * @param res The residual vector.
 * @param R The measurement noise covariance matrix.
 * @param P_updated The updated covariance matrix (output).
 * @param x_updated The updated state vector (output).
 */
void StateHelper::HuberFilter(std::shared_ptr<State> state,
                const Eigen::MatrixXd &H,
                const Eigen::VectorXd &res,
                const Eigen::MatrixXd &R,
                Eigen::MatrixXd &P_updated,
                Eigen::VectorXd &x_updated)
{
    auto rT0_1 = boost::posix_time::microsec_clock::local_time();
    // 获取先验协方差
    const Eigen::MatrixXd P_minus = StateHelper::get_full_covariance(state);
    // 构造S_k矩阵
    const int measurement_size = res.size();
    const int state_size =  state->max_covariance_size();
    Eigen::MatrixXd S_k = Eigen::MatrixXd::Zero(measurement_size + state_size, measurement_size + state_size);
    S_k.block(0, 0, measurement_size, measurement_size) = R;
    S_k.block(measurement_size, measurement_size, state_size, state_size) = P_minus;
    // 计算S_k^(-1/2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(S_k);
    Eigen::MatrixXd S_k_inv_sqrt = eigen_solver.operatorInverseSqrt();

    // 构造z_k和M_k
    Eigen::VectorXd z_k(measurement_size + state_size);
    z_k.head(measurement_size) = res;
    z_k.tail(state_size).setZero();
    z_k = S_k_inv_sqrt * z_k;
    Eigen::MatrixXd M_k(measurement_size + state_size, state_size);
    M_k.topRows(measurement_size) = H;
    M_k.bottomRows(state_size) = Eigen::MatrixXd::Identity(state_size, state_size);
    M_k = S_k_inv_sqrt * M_k;
    
     // Huber迭代求解
    const double gamma = 1.345; 
    const int max_iterations = 20;
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(state_size);

    
    // 初始解使用普通最小二乘
    Eigen::MatrixXd Psi = Eigen::MatrixXd::Identity(z_k.size(), z_k.size());
    
    for(int iter = 0; iter < max_iterations; iter++) {
        // 求解加权最小二乘问题
         Eigen::VectorXd delta_x_new = (M_k.transpose() * Psi * M_k).ldlt().solve(M_k.transpose() * Psi * z_k);
        
        // 计算残差
        Eigen::VectorXd zeta = M_k * delta_x - z_k;
        
        // 更新Huber权重
        Eigen::MatrixXd Psi_new = Eigen::MatrixXd::Identity(zeta.size(), zeta.size());
        for(int i = 0; i < zeta.size(); i++) {
            double abs_zeta = std::abs(zeta(i));
            if(abs_zeta > gamma) {
                Psi_new(i,i) = gamma / abs_zeta;
            }
        }
        
        // 检查收敛
        if((delta_x_new - delta_x).norm() < 1e-6) {
            delta_x = delta_x_new;
            Psi = Psi_new;
            break;
        }

        delta_x = delta_x_new;
        Psi = Psi_new;
    }
    
    // 使用最终的权重矩阵计算增益K
    Eigen::MatrixXd Psi_y = Psi.topLeftCorner(measurement_size, measurement_size);
    Eigen::MatrixXd Psi_x = Psi.bottomRightCorner(state_size, state_size);
    
    Eigen::MatrixXd P_sqrt = P_minus.llt().matrixL();
    Eigen::MatrixXd R_sqrt = R.llt().matrixL();
    
    Eigen::MatrixXd K = P_sqrt * Psi_x.inverse() * P_sqrt.transpose() * H.transpose() * 
                       (H * P_sqrt * Psi_x.inverse() * P_sqrt.transpose() * H.transpose() + 
                        R_sqrt * Psi_y.inverse() * R_sqrt.transpose()).inverse();
    
    // 计算最终的状态更新
    x_updated = K * res;
/*      for (size_t i = 0; i < state->_variables.size(); i++) {
        state->_variables.at(i)->update(
            x_updated.segment(state->_variables.at(i)->id(), state->_variables.at(i)->size()));
    } */
        // 保存 IMU 位置信息
/*     Eigen::Vector3d imu_position = state->_imu->pos();
    std::ofstream huber_pos_file("/home/zhou/workspace/data/result/huber_position.txt", std::ios_base::app);
    huber_pos_file << std::setprecision(15)
                   << state->_timestamp << " "
                   << imu_position.transpose() << std::endl;
    huber_pos_file.close(); */
    
    // 更新协方差
    const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(state_size, state_size) - K * H;
    //P_updated = I_KH * P_minus * I_KH.transpose() + K * R * K.transpose(); 
    P_updated = (Eigen::MatrixXd::Identity(state_size, state_size) - K * H) *P_minus * Psi_x.inverse();
    P_updated = P_updated.eval().selfadjointView<Eigen::Upper>();
    LimitMinDiagValue(1e-12, &P_updated);
    auto rT0_2 = boost::posix_time::microsec_clock::local_time();
        double time_total1 = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
 
    std::cout << "HUBER: " << time_total1 << std::endl;
    //state->_Cov = P_updated;
}




void StateHelper::ChiSquareFilter(std::shared_ptr<State> state,
                                  const Eigen::MatrixXd &H,
                                  const Eigen::VectorXd &res,
                                  const Eigen::MatrixXd &R,
                                  Eigen::MatrixXd &P_updated,
                                  Eigen::VectorXd &x_updated)
{ auto rT0_1 = boost::posix_time::microsec_clock::local_time();
    // 获取先验协方差矩阵和状态维度
    Eigen::MatrixXd P_minus = StateHelper::get_full_covariance(state);
    int state_size = state->max_covariance_size();
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(state_size);

    // 卡方检验参数设置
    int degrees_of_freedom = res.size();
    double confidence_level = 0.99;
    double chi_square_threshold = ChiSquareThreshold(degrees_of_freedom, confidence_level);

    // 初始化或更新测量噪声协方差矩阵
    static Eigen::MatrixXd R_adaptive = R; // 初始值为传入的 R
    static double previous_inflation_factor = 1.0; // 上一次的膨胀因子

    // 计算卡尔曼增益所需的中间变量
    Eigen::MatrixXd PHt = P_minus * H.transpose();
    Eigen::MatrixXd S = H * P_minus * H.transpose() + R_adaptive;
    
    // 数值稳定性处理，防止 S 矩阵奇异
    S += 1e-8 * Eigen::MatrixXd::Identity(S.rows(), S.cols());

    // 使用 Cholesky 分解求解逆矩阵
    Eigen::MatrixXd S_inv = S.llt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    Eigen::MatrixXd K = PHt * S_inv;

    // 计算卡方统计量
    double chi_square = res.transpose() * S_inv * res;

   /*  // 调试信息
    std::cout << "Chi-square value: " << chi_square << std::endl;
    std::cout << "Chi-square threshold: " << chi_square_threshold << std::endl; */

    double inflation_factor;
    double max_inflation_factor = 20.0; // 限制最大膨胀因子
    double min_inflation_factor = 1.0;  // 限制最小膨胀因子

    if (chi_square <= chi_square_threshold) {
        // 正常更新

        // 平滑恢复测量噪声协方差矩阵
        double beta = 0.95; // 平滑系数
        R_adaptive = beta * R_adaptive + (1 - beta) * R;

        // 重置膨胀因子
        inflation_factor = 1.0;
    } else {
        // 检测到异常

        // 计算膨胀因子，加入平滑
        double alpha = 0.95; // 平滑系数
        inflation_factor = alpha * previous_inflation_factor + (1 - alpha) * (chi_square / chi_square_threshold);

        // 限制膨胀因子范围
        inflation_factor = std::min(std::max(inflation_factor, min_inflation_factor), max_inflation_factor);

        // 更新测量噪声协方差矩阵
        R_adaptive = R * inflation_factor;
    }

    // 更新膨胀因子
    previous_inflation_factor = inflation_factor;

    // 调试信息
 /*    std::cout << "Inflation factor: " << inflation_factor << std::endl;
    std::cout << "R:" << std::endl;
    std::cout << R << std::endl;
    std::cout << "R_adaptive:" << std::endl;
    std::cout << R_adaptive << std::endl; */

    // 重新计算卡尔曼增益
    S = H * P_minus * H.transpose() + R_adaptive;
    S += 1e-8 * Eigen::MatrixXd::Identity(S.rows(), S.cols()); // 数值稳定性处理
    S_inv = S.llt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    K = PHt * S_inv;

    // 更新状态估计
    Eigen::VectorXd dx = K * res;
   

    // 更新协方差矩阵
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(state_size, state_size) - K * H;
    P_updated = I_KH * P_minus * I_KH.transpose() + K * R_adaptive * K.transpose();

   
   

    // 更新状态
    for (size_t i = 0; i < state->_variables.size(); i++) {
        state->_variables.at(i)->update(
            dx.segment(state->_variables.at(i)->id(), state->_variables.at(i)->size()));
    }
        // 保存 IMU 位置信息
    Eigen::Vector3d imu_position = state->_imu->pos();
    std::ofstream chi2_pos_file("/home/zhou/workspace/data/result/chi2_position.txt", std::ios_base::app);
    chi2_pos_file << std::setprecision(15)
                  << state->_timestamp << " "
                  << imu_position.transpose() << std::endl;
    chi2_pos_file.close();
    x_updated = dx;
  P_updated = 0.5 * (P_updated + P_updated.transpose());
  LimitMinDiagValue(1e-12, &P_updated); 
 // state->_Cov = P_updated;
   

    // 限制协方差矩阵的最小对角线值
    StateHelper::LimitMinDiagValue(1e-8, &P_updated);


    auto rT0_2 = boost::posix_time::microsec_clock::local_time();
    double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
    std::cout << "ChiSquare time: " << time_total << std::endl;
}


// 计算给定自由度和置信水平的卡方阈值
double StateHelper::ChiSquareThreshold(int degrees_of_freedom, double confidence_level)
{
    // 使用近似算法计算卡方阈值
    // 这里使用 Wilson-Hilferty 近似
    if (confidence_level <= 0.0 || confidence_level >= 1.0 || degrees_of_freedom <= 0) {
        std::cerr << "Invalid input parameters for ChiSquareThreshold." << std::endl;
        return 0.0;
    }

    // 使用逆正态分布函数
    double z = InverseNormalCDF(confidence_level);
    double k = static_cast<double>(degrees_of_freedom);
    double chi_square_threshold = k * pow(1.0 - 2.0 / (9.0 * k) + z * sqrt(2.0 / (9.0 * k)), 3);

    return chi_square_threshold;
}

// 逆正态分布函数的近似实现
double StateHelper::InverseNormalCDF(double p)
{
    // 检查输入参数
    if (p <= 0.0 || p >= 1.0) {
        std::cerr << "Invalid input for InverseNormalCDF." << std::endl;
        return 0.0;
    }

    // 使用 Abramowitz and Stegun 提供的近似公式
    // 常数定义（Abramowitz and Stegun, 1964）
    const double a1 = -39.69683028665376;
    const double a2 = 220.9460984245205;
    const double a3 = -275.9285104469687;
    const double a4 = 138.3577518672690;
    const double a5 = -30.66479806614716;
    const double a6 = 2.506628277459239;

    const double b1 = -54.47609879822406;
    const double b2 = 161.5858368580409;
    const double b3 = -155.6989798598866;
    const double b4 = 66.80131188771972;
    const double b5 = -13.28068155288572;

    const double c1 = -0.007784894002430293;
    const double c2 = -0.3223964580411365;
    const double c3 = -2.400758277161838;
    const double c4 = -2.549732539343734;
    const double c5 = 4.374664141464968;
    const double c6 = 2.938163982698783;

    const double d1 = 0.007784695709041462;
    const double d2 = 0.3224671290700398;
    const double d3 = 2.445134137142996;
    const double d4 = 3.754408661907416;

    // 定义概率界限
    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r, z;

    if (p < p_low) {
        // 适用于极小概率的近似公式
        q = std::sqrt(-2.0 * std::log(p));
        z = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6;
        z = z / (((((d1 * q + d2) * q + d3) * q + d4) * q) + 1.0);
    } else if (p <= p_high) {
        // 适用于中心概率的近似公式
        q = p - 0.5;
        r = q * q;
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q;
        z = z / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        // 适用于极大概率的近似公式
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6);
        z = z / (((((d1 * q + d2) * q + d3) * q + d4) * q) + 1.0);
    }


    // 返回计算得到的 z 值
    return z;
}




void StateHelper::set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                         const std::vector<std::shared_ptr<ov_type::Type>> &order) {

  // We need to loop through each element and overwrite the current covariance values
  // For example consider the following:
  // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
  // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
  //     [ P_po P_pp ]        [  0    P*    0  ]
  //                          [ P_po  0   P_pp ]
  // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
  // This is normally the care on startup (for example between calibration and the initial state

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
}

Eigen::MatrixXd StateHelper::get_marginal_covariance(std::shared_ptr<State> state,
                                                     const std::vector<std::shared_ptr<Type>> &small_variables) {

  // Calculate the marginal covariance size we need to make our matrix
  int cov_size = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    cov_size += small_variables[i]->size();
  }

  // Construct our return covariance
  Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < small_variables.size(); k++) {
      Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
          state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
      k_index += small_variables[k]->size();
    }
    i_index += small_variables[i]->size();
  }

  // Return the covariance
  // Small_cov = 0.5*(Small_cov+Small_cov.transpose());
  return Small_cov;
}

Eigen::MatrixXd StateHelper::get_full_covariance(std::shared_ptr<State> state) {

  // Size of the covariance is the active
  int cov_size = (int)state->_Cov.rows();

  // Construct our return covariance
  Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // Copy in the active state elements
  full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov;

  // Return the covariance
  return full_cov;
}

void StateHelper::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {

  // Check if the current state has the element we want to marginalize
  if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
    PRINT_ERROR(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
  //
  //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
  //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
  //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
  //
  //  to
  //
  //  P_(x_1,x_1) P(x_1,x_2)
  //  P_(x_2,x_1) P(x_2,x_2)
  //
  // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

  int marg_size = marg->size();
  int marg_id = marg->id();
  int x2_size = (int)state->_Cov.rows() - marg_id - marg_size;

  Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);

  // P_(x_1,x_1)
  Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

  // P_(x_1,x_2)
  Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

  // P_(x_2,x_1)
  Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

  // P(x_2,x_2)
  Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);

  // Now set new covariance
  // state->_Cov.resize(Cov_new.rows(),Cov_new.cols());
  state->_Cov = Cov_new;
  // state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
  assert(state->_Cov.rows() == Cov_new.rows());

  // Now we keep the remaining variables and update their ordering
  // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
  std::vector<std::shared_ptr<Type>> remaining_variables;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // Only keep non-marginal states
    if (state->_variables.at(i) != marg) {
      if (state->_variables.at(i)->id() > marg_id) {
        // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
        state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
      }
      remaining_variables.push_back(state->_variables.at(i));
    }
  }

  // Delete the old state variable to free up its memory
  // NOTE: we don't need to do this any more since our variable is a shared ptr
  // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
  // delete marg;
  marg->set_local_id(-1);

  // Now set variables as the remaining ones
  state->_variables = remaining_variables;
}

std::shared_ptr<Type> StateHelper::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone) {

  // Get total size of new cloned variables, and the old covariance size
  int total_size = variable_to_clone->size();
  int old_size = (int)state->_Cov.rows();
  int new_loc = (int)state->_Cov.rows();

  // Resize both our covariance to the new size
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));

  // What is the new state, and variable we inserted
  const std::vector<std::shared_ptr<Type>> new_variables = state->_variables;
  std::shared_ptr<Type> new_clone = nullptr;

  // Loop through all variables, and find the variable that we are going to clone
  for (size_t k = 0; k < state->_variables.size(); k++) {

    // Skip this if it is not the same
    // First check if the top level variable is the same, then check the sub-variables
    std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
    if (state->_variables.at(k) == variable_to_clone) {
      type_check = state->_variables.at(k);
    } else if (type_check != variable_to_clone) {
      continue;
    }

    // So we will clone this one
    int old_loc = type_check->id();

    // Copy the covariance elements
    state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
    state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
    state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);

    // Create clone from the type being cloned
    new_clone = type_check->clone();
    new_clone->set_local_id(new_loc);
    break;
  }

  // Check if the current state has this variable
  if (new_clone == nullptr) {
    PRINT_ERROR(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Add to variable list and return
  state->_variables.push_back(new_clone);
  return new_clone;
}

bool StateHelper::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                             Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {

  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // First we perform QR givens to seperate the system
  // The top will be a system that depends on the new state, while the bottom does not
  size_t new_var_size = new_variable->size();
  assert((int)new_var_size == H_L.cols());

  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_L.cols(); ++n) {
    for (int m = (int)H_L.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // Separate into initializing and updating portions
  // 1. Invertible initializing system
  Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
  Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
  Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
  Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

  // 2. Nullspace projected updating system
  Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
  Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
  Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

  //==========================================================
  //==========================================================

  // Do mahalanobis distance testing
  Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
  assert(Rup.rows() == Hup.rows());
  assert(Hup.cols() == P_up.cols());
  Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
  double chi2 = resup.dot(S.llt().solve(resup));

  // Get what our threshold should be
  boost::math::chi_squared chi_squared_dist(res.rows());
  double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  if (chi2 > chi_2_mult * chi2_check) {
    return false;
  }

  //==========================================================
  //==========================================================
  // Finally, initialize it in our state
  StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);

  // Update with updating portion
  if (Hup.rows() > 0) {
    StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup);
  }
  return true;
}

void StateHelper::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                        const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H_R,
                                        const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {

  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H_L.rows() == res.rows());
  assert(H_L.rows() == H_R.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order.at(i);
      M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
             H_R.block(0, H_id[i], H_R.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }

  //==========================================================
  //==========================================================
  // Get covariance of this small jacobian
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // M = H_R*Cov*H_R' + R
  Eigen::MatrixXd M(H_R.rows(), H_R.rows());
  M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
  M.triangularView<Eigen::Upper>() += R;

  // Covariance of the variable/landmark that will be initialized
  assert(H_L.rows() == H_L.cols());
  assert(H_L.rows() == new_variable->size());
  Eigen::MatrixXd H_Linv = H_L.inverse();
  Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();

  // Augment the covariance matrix
  size_t oldSize = state->_Cov.rows();
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
  state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -M_a * H_Linv.transpose();
  state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
  state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;

  // Update the variable that will be initialized (invertible systems can only update the new variable).
  // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
  new_variable->update(H_Linv * res);

  // Now collect results, and add it to the state variables
  new_variable->set_local_id(oldSize);
  state->_variables.push_back(new_variable);

  // std::stringstream ss;
  // ss << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;
  // PRINT_DEBUG(ss.str().c_str());
}

void StateHelper::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w) {

  // We can't insert a clone that occured at the same timestamp!
  if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
    PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our cloner and add it to our vector of types
  // NOTE: this will clone the clone pose to the END of the covariance...
  std::shared_ptr<Type> posetemp = StateHelper::clone(state, state->_imu->pose());

  // Cast to a JPL pose type, check if valid
  std::shared_ptr<PoseJPL> pose = std::dynamic_pointer_cast<PoseJPL>(posetemp);
  if (pose == nullptr) {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Append the new clone to our clone vector
  state->_clones_IMU[state->_timestamp] = pose;

  // If we are doing time calibration, then our clones are a function of the time offset
  // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
  // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
  if (state->_options.do_calib_camera_timeoffset) {
    // Jacobian to augment by
    Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = last_w;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    // Augment covariance with time offset Jacobian
    // TODO: replace this with a call to the EKFPropagate function instead....
    state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
        state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
    state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
        dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
  }
}

void StateHelper::marginalize_old_clone(std::shared_ptr<State> state) {
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    double marginal_time = state->margtimestep();
    // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
    std::lock_guard<std::mutex> lock(state->_mutex_state);
    assert(marginal_time != INFINITY);
    StateHelper::marginalize(state, state->_clones_IMU.at(marginal_time));
    // Note that the marginalizer should have already deleted the clone
    // Thus we just need to remove the pointer to it from our state
    state->_clones_IMU.erase(marginal_time);
  }
}

void StateHelper::marginalize_slam(std::shared_ptr<State> state) {
  // Remove SLAM features that have their marginalization flag set
  // We also check that we do not remove any aruoctag landmarks
  int ct_marginalized = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((*it0).second->should_marg && (int)(*it0).first > 4 * state->_options.max_aruco_features) {
      StateHelper::marginalize(state, (*it0).second);
      it0 = state->_features_SLAM.erase(it0);
      ct_marginalized++;
    } else {
      it0++;
    }
  }
}