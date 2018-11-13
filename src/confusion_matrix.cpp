#include "confusion_matrix.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

std::ostream& operator<<( std::ostream& os, const Confusion_Matrix& cm )
{
	auto p = os.precision();

	os << "Confusion Matrix:\n";
	os << std::right << std::setw(10) << cm.TP() << ", " << std::setw(10) << cm.FP() << '\n';
	os << std::right << std::setw(10) << cm.FN() << ", " << std::setw(10) << cm.TN() << '\n';
	os << "total population   \t: " << std::left << cm.total_population() << '\n';
	os << "condition positive \t: " << std::left << cm.condition_positive() << '\n';
	os << "condition negative \t: " << std::left << cm.condition_negative() << '\n';
	os << "predicted positive \t: " << std::left << cm.predicted_positive() << '\n';
	os << "predicated negative\t: " << std::left << cm.predicted_negative() << '\n';
	os.precision(8);
	os.setf( std::ios::fixed, std::ios::floatfield );
	os << "TPR (sensitivity)    \t: " << cm.TPR()        << "\tTNR (specificity)         : " << cm.TNR() << '\n';
	os << "FPR (fall-out)       \t: " << cm.FPR()        << "\tFNR (miss rate)           : " << cm.FNR() << '\n';
	os << "PPV (precision)      \t: " << cm.PPV()        << "\tNPV (neg pred value)      : " << cm.NPV() << '\n';
	os << "FDR (false disc rate)\t: " << cm.FDR()        << "\tFOR (false omission rate) : " << cm.FOR() << '\n';
	os << "MCC (Matthews CC)    \t: " << cm.MCC()        << "\tinformedness              : " << cm.informedness() << '\n';
	os << "Accuracy             \t: " << cm.accuracy()   << "\tPrevalence                : " << cm.prevalence() << '\n';
	os << "markedness           \t: " << cm.markedness() << "\tF1                        : " << cm.F1() << '\n';
	os << "Pos Likelihood Ratio \t: " << cm.LR_pos()     << "\tNeg Likelihood Ratio      : " << cm.LR_neg() << '\n';
	os << "DOR (Diag Odds Ratio)\t: " << cm.DOR() << '\n';
	os.precision( p );
	return os;
}


Confusion_Matrix::Confusion_Matrix( const Eigen::Matrix2i& cm )
    : cm_{ cm.template cast<uint64_t>() }
{}

Confusion_Matrix::Confusion_Matrix( const std::vector<int>& y_true, const std::vector<int>& y_pred )
{
    build ( Eigen::VectorXi::Map( y_true.data(), y_true.size() ), Eigen::VectorXi::Map( y_pred.data(), y_pred.size() ) );
}

/**
 * Assumes 1 is positive class.
 * Assumes 0 is negative class.
 */
Confusion_Matrix::Confusion_Matrix( const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred )
{
    build( y_true, y_pred );
}

Confusion_Matrix::~Confusion_Matrix()
{}

void Confusion_Matrix::build( const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred )
{
    // element-wise addition and subtraction.
    // having the signed integers in the result vectors is IMPORTANT!
    Eigen::VectorXi add = y_true.array() + y_pred.array();
    Eigen::VectorXi sub = y_true.array() - y_pred.array();

    // cool Eigen boolean ops on arrays.
    uint64_t true_positives = ( add.array() == 2 ).count();  // both have 1's
    uint64_t true_negatives = ( add.array() == 0 ).count();  // both have 0's
    uint64_t false_positives = ( sub.array() < 0 ).count();  // 0 - 1
    uint64_t false_negatives = ( sub.array() > 0 ).count();  // 1 - 0

    // arrange based on wikipedia.
    //                          TRUE CONDITION 
    //  PREDICTED CONDITION     positive,     negative
    //             positive     TP            FP
    //             negative     FN            TN
    cm_ << true_positives, false_positives, 
           false_negatives, true_negatives;
}

const Eigen::Matrix<uint64_t, 2, 2>& Confusion_Matrix::get( void ) const
{
    return cm_;
}

uint64_t Confusion_Matrix::total_population( void ) const
{
    return cm_.sum();
}

uint64_t Confusion_Matrix::predicted_positive( void ) const
{
    return cm_.row(0).sum();
}

uint64_t Confusion_Matrix::predicted_negative( void ) const
{
    return cm_.row(1).sum();
}

uint64_t Confusion_Matrix::condition_positive( void ) const
{
    return cm_.col(0).sum();
}

uint64_t Confusion_Matrix::condition_negative( void ) const
{
    return cm_.col(1).sum();
}

uint64_t Confusion_Matrix::true_positives( void ) const
{
    return cm_(0,0);
}

uint64_t Confusion_Matrix::TP( void ) const
{
    return cm_(0,0);
}

uint64_t Confusion_Matrix::true_negatives( void ) const
{
    return cm_(1,1);
}

uint64_t Confusion_Matrix::TN( void ) const
{
    return cm_(1,1);
}

uint64_t Confusion_Matrix::false_positives( void ) const
{
    return cm_(0,1);
}

uint64_t Confusion_Matrix::FP( void ) const
{
    return cm_(0,1);
}

uint64_t Confusion_Matrix::false_negatives( void ) const
{
    return cm_(1,0);
}

uint64_t Confusion_Matrix::FN( void ) const
{
    return cm_(1,0);
}


double Confusion_Matrix::TPR( void ) const
{
	return static_cast<double>(TP())/static_cast<double>(condition_positive());
}

double Confusion_Matrix::true_positive_rate( void ) const
{
	return TPR();
}

double Confusion_Matrix::sensitivity( void ) const
{
	return TPR();
}

double Confusion_Matrix::recall( void ) const
{
	return TPR();
}

double Confusion_Matrix::TNR( void ) const
{
	return static_cast<double>(TN())/static_cast<double>(condition_negative());
}

double Confusion_Matrix::true_negative_rate( void ) const
{
	return TNR();
}

double Confusion_Matrix::specificity( void ) const
{
	return TNR();
}

double Confusion_Matrix::PPV( void ) const
{
	return static_cast<double>(TP())/static_cast<double>(predicted_positive());
}

double Confusion_Matrix::positive_predictive_value( void ) const
{
	return PPV();
}

double Confusion_Matrix::precision( void ) const
{
	return PPV();
}

double Confusion_Matrix::NPV( void ) const
{
	return static_cast<double>(TN())/static_cast<double>(predicted_negative());
}

double Confusion_Matrix::negative_predicative_value( void ) const
{
	return NPV();
}

double Confusion_Matrix::FNR( void ) const
{
	return static_cast<double>(FN())/static_cast<double>(condition_positive());
}

double Confusion_Matrix::false_negative_rate( void ) const
{
	return FNR();
}

double Confusion_Matrix::miss_rate( void ) const
{
	return FNR();
}

double Confusion_Matrix::FPR( void ) const
{
	return static_cast<double>(FP())/static_cast<double>(condition_negative());
}

double Confusion_Matrix::false_positive_rate( void ) const
{
	return FPR();
}

double Confusion_Matrix::fall_out( void ) const
{
	return FPR();
}

double Confusion_Matrix::FDR( void ) const
{
	return 1.0-PPV();
}

double Confusion_Matrix::false_discovery_rate( void ) const
{
	return FDR();
}

double Confusion_Matrix::FOR( void ) const
{
	return 1.0-NPV();
}

double Confusion_Matrix::false_omission_rate( void ) const
{
	return FOR();
}

double Confusion_Matrix::accuracy( void ) const
{
	return static_cast<double>((TP() + TN())) / static_cast<double>(total_population());
}

double Confusion_Matrix::F1( void ) const
{
    return 2.0 / ( 1.0 / recall() + 1.0 / precision() );
}

double Confusion_Matrix::MCC( void ) const
{
    double tp = static_cast<double>( TP() );
    double tn = static_cast<double>( TN() );
    double fp = static_cast<double>( FP() );
    double fn = static_cast<double>( FN() );
	return ( tp * tn + fp * fn ) / std::sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn));
}

double Confusion_Matrix::mathews_correlation_coefficient( void ) const
{
	return MCC();
}

double Confusion_Matrix::informedness( void ) const
{
	return TPR() + TNR() - 1.0;
}

double Confusion_Matrix::markedness( void ) const
{
	return PPV() + NPV() - 1.0;
}

double Confusion_Matrix::prevalence( void ) const
{
    return static_cast<double>( condition_positive() ) / static_cast<double>( total_population() );
}

double Confusion_Matrix::positive_likelihood_ratio( void ) const
{
    return LR_pos();
}

double Confusion_Matrix::LR_pos( void ) const
{
    return TPR()/FPR();
}

double Confusion_Matrix::negative_likelihood_ratio( void ) const
{
    return LR_neg();
}

double Confusion_Matrix::LR_neg( void ) const
{
    return FNR()/TNR();
}

double Confusion_Matrix::diagnostic_odds_ratio( void ) const
{
    return DOR(); 
}

double Confusion_Matrix::DOR( void ) const
{
    return LR_pos()/LR_neg();
}

