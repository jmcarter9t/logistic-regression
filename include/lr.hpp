#ifndef __LR_H__
#define __LR_H__

#include "Core"
#include "Eigen"
#include <string>
#include <vector>
#include <random>

#include "utilities.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/file_sinks.h"
#include "spdlog/fmt/ostr.h"

class LR {
    public:

        static double sigmoid(double x);
        static void sigmoid( Eigen::MatrixXd& m );
        static void sigmoid( Eigen::VectorXd& v );
        static double cross_entropy( double y, double y_hat );
        static double cross_entropy_loss( const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat );

        /**
         * Sets things up so you can fit data and discover weights.
         */
        LR(uint32_t maximum_iterations, double loss_threshold, double learning_rate = 1.0 );
        LR( const StrVector& feature_names, uint32_t maximum_iterations, double loss_threshold, double learning_rate = 1.0 );

        /**
         * Sets things up so you can just classify using pre-discovered weights.
         * If the feature_names vector is empty, these will be made based on the length of weights.
         */
        LR( const StrVector& feature_names, const std::vector<double>& weights, double classification_threshold );
        LR( const StrVector& feature_names, Eigen::VectorXd& weights, double classification_threshold );
        LR( const std::string& wfile, double classification_threshold );

        ~LR();

        // Deprecated
        void fit_non_irls( Eigen::MatrixXd& X, const Eigen::VectorXd& Y);
        void fit_non_irls( Eigen::MatrixXd&& X, const Eigen::VectorXd&& Y);

        // These fit using iterative reweighted least squares -- better than the above.
        void fit( Eigen::MatrixXd& X, const Eigen::VectorXd& Y );
        void fit( Eigen::MatrixXd&& X, const Eigen::VectorXd&& Y );

        Eigen::VectorXd predict_sigmoids( const Eigen::MatrixXd& X_in ) const;
        double predict_sigmoid( const Eigen::Matrix<double,1,Eigen::Dynamic>& V_in ) const;
        double predict_sigmoid( const std::vector<double>& V_in ) const;
        Eigen::VectorXi predict(Eigen::MatrixXd& X_in) const;

        int classify( double pred_y ) const;
        int classify( const Eigen::Matrix<double,1,Eigen::Dynamic>& V_in ) const;
        int classify( const std::vector<double>& V_in ) const;

        Eigen::VectorXd get_random_weights( uint32_t size );
        const Eigen::VectorXd& get_model_weights() const;
        StrVector get_feature_names() const;

        void saveWeights(const std::string& filename);
        void loadWeights(const std::string& filename);

    private:

        std::shared_ptr<spdlog::logger> clog;

        // to randomize weights at start.
        std::random_device rd_;
        std::mt19937_64 rng_;
        std::uniform_real_distribution<> dist_;

        StrVector feature_names_;
        Eigen::VectorXd weights_;                           ///< This is a column vector (usually a row vector in literature); these are the "model parameters"
                                                            ///< ( bias term, parm for f1, parm for f2, ..., parm for fn )

        uint32_t maximum_iterations_{ 100 };                ///< max iterations to converge.
        double loss_threshold_{ 0.05 };                     ///< error threshold_ence
        double classification_threshold_{ 0.5 };            ///< error threshold_ence
        double learning_rate_{ 0.01 };                      ///< learning rate.

        void build_feature_names_( int num_features );
};

#endif
