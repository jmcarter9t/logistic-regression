#ifndef __CONFUSION_MATRIX_H__
#define __CONFUSION_MATRIX_H__

#include "Eigen"

class Confusion_Matrix {
    public:
        Confusion_Matrix( const Eigen::Matrix2i& cm );
        Confusion_Matrix( const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred );
        Confusion_Matrix( const std::vector<int>& y_true, const std::vector<int>& y_pred );

        ~Confusion_Matrix();

        const Eigen::Matrix<uint64_t, 2, 2>& get( void ) const;
        uint64_t total_population( void ) const;
        uint64_t predicted_positive( void ) const;
        uint64_t predicted_negative( void ) const;
        uint64_t condition_positive( void ) const;
        uint64_t condition_negative( void ) const;
        uint64_t TP( void ) const;
        uint64_t true_positives( void ) const;
        uint64_t TN( void ) const;
        uint64_t true_negatives( void ) const;
        uint64_t FP( void ) const;
        uint64_t false_positives( void ) const;
        uint64_t FN( void ) const;
        uint64_t false_negatives( void ) const;
        double TPR( void ) const;
        double true_positive_rate( void ) const;
        double sensitivity( void ) const;
        double recall( void ) const;
        double TNR( void ) const;
        double true_negative_rate( void ) const;
        double specificity( void ) const;
        double PPV( void ) const;
        double positive_predictive_value( void ) const;
        double precision( void ) const;
		double NPV( void ) const;
		double negative_predicative_value( void ) const;
        double FNR( void ) const;
        double false_negative_rate( void ) const;
        double miss_rate( void ) const;
        double FPR( void ) const;
        double false_positive_rate( void ) const;
        double fall_out( void ) const;
        double FDR( void ) const;
        double false_discovery_rate( void ) const;
        double FOR( void ) const;
        double false_omission_rate( void ) const;
        double accuracy( void ) const;
        double prevalence( void ) const;
        double F1( void ) const;
        double MCC( void ) const;
        double mathews_correlation_coefficient( void ) const;
        double informedness( void ) const;
        double markedness( void ) const;
        double positive_likelihood_ratio( void ) const;
        double LR_pos( void ) const;
        double negative_likelihood_ratio( void ) const;
        double LR_neg( void ) const;
        double diagnostic_odds_ratio( void ) const;
        double DOR( void ) const;
		friend std::ostream& operator<<( std::ostream& os, const Confusion_Matrix& cm );

    private:
        Eigen::Matrix<uint64_t,2,2> cm_;
        void build( const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred );
};

#endif
