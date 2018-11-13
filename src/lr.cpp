#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>

#include "utilities.hpp"
#include "QR"    
#include "lr.hpp"

// STATIC METHODS

double LR::sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
}

/**
 * The Eigen way.
 */
void LR::sigmoid( Eigen::MatrixXd& m )
{
    m = inverse( 1.0 + exp( ( -1.0 * m ).array() ) );
}

/**
 * The Eigen way.
 */
void LR::sigmoid( Eigen::VectorXd& v )
{
    v = inverse( 1.0 + exp( ( -1.0 * v ).array() ) );
}

/**
 * This one doesn't rely on the cross_entropy helper function.
 * Would be interesting to benchmark.
 */
double LR::cross_entropy_loss( const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat )
{
    // This is the Eigen way. Use array to get component level ops.
    Eigen::VectorXd class_a = y.array() * y_hat.array().unaryExpr( []( double x ){ return std::log2(x); } );
    Eigen::VectorXd class_b = ( 1.0 - y.array() ) * y_hat.array().unaryExpr( []( double x ){ return std::log2(1.0-x); } );
    return -1.0 / y.size() * (class_a + class_b).sum();
}

double LR::cross_entropy( double y, double y_hat )
{
    // return class A cost - class B cost;
    return (y * log2( y_hat )) + (( 1.0 - y ) * log2( 1.0 - y_hat ));
}

// INSTANCE METHODS

/**
 *
 * @param feature_names DOES NOT include a name for the bias term (BIAS).
 */
LR::LR( const StrVector& feature_names, uint32_t maximum_iterations, double loss_threshold, double learning_rate )
    : maximum_iterations_{ maximum_iterations }
    , learning_rate_{ learning_rate }
    , loss_threshold_{ loss_threshold }
    , rng_{ rd_() }
    , dist_{-1.0, 1.0}  // random weights between [0,1.0)
    , feature_names_{ feature_names }  // random weights between [0,1.0)
    , clog{ spdlog::get( "clog" ) }
{
    feature_names_.insert( feature_names_.begin(), "BIAS" );
}

/**
 *
 * @param feature_names DOES NOT include a name for the bias term (BIAS).
 */
LR::LR(uint32_t maximum_iterations, double loss_threshold, double learning_rate )
    : maximum_iterations_{ maximum_iterations }
    , learning_rate_{ learning_rate }
    , loss_threshold_{ loss_threshold }
    , rng_{ rd_() }
    , dist_{-1.0, 1.0}  // random weights between [0,1.0)
    , clog{ spdlog::get( "clog" ) }
{
    feature_names_.insert( feature_names_.begin(), "BIAS" );
}

/**
 * The file should be a comma delimited file with the first element being the name of the feature
 * and the second being the weight.
 */
LR::LR( const std::string& wfile, double classification_threshold )
    : classification_threshold_{ classification_threshold }
    , clog{ spdlog::get( "clog" ) }
{
    loadWeights( wfile );       // throws
}
        
/**
 *
 * @param feature_names DOES NOT include a name for the bias term (BIAS).
 */
LR::LR( const StrVector& feature_names, Eigen::VectorXd& weights, double classification_threshold )
    : feature_names_{ feature_names }
    , weights_{ weights }
    , classification_threshold_{ classification_threshold }
    , clog{ spdlog::get( "clog" ) }
{
    feature_names_.insert( feature_names_.begin(), "BIAS" );
    if ( feature_names_.empty() || feature_names_.size() != weights_.size() ) {
        throw std::invalid_argument("feature names size and weights size are not the same." );
    } 
}
        
/**
 * @param feature_names DOES NOT include a name for the bias term (BIAS).
 */
LR::LR( const StrVector& feature_names, const std::vector<double>& weights, double classification_threshold )
    : feature_names_{ feature_names }
    , weights_{ weights.size() }
    , classification_threshold_{ classification_threshold }
    , clog{ spdlog::get( "clog" ) }
{
    feature_names_.insert( feature_names_.begin(), "BIAS" );
    if ( feature_names_.empty() || feature_names_.size() != weights.size() ) {
        throw std::invalid_argument("feature names size and weights size are not the same." );
    } 

    for ( int i=0; i < weights.size(); ++i )  {
        weights_(i) = weights[i];
    }
}

LR::~LR(){}

Eigen::VectorXd LR::get_random_weights( uint32_t size ) 
{
    std::vector<double> v( size );
    generate( begin( v ), end( v ), [this]() { return dist_( rng_ ); } );
    return Eigen::VectorXd::Map( v.data(), v.size() );
}

/**
 * Placeholder or surrogate feature names in lieu of them being specified.
 */
void LR::build_feature_names_( int num_features )
{
    feature_names_.clear();
    for ( std::size_t i = 0; i < num_features; ++i ) {
        feature_names_.push_back( std::to_string( i ) );
        clog->trace("build feature name: {}", feature_names_.back());
    }
}

/**
 * Fit data matrix X and labels Y to a logistic regression model using the set learning parameters.
 *
 * This instance of LR will have its internal weights set correctly.
 *
 * @param X the data matrix ( data instances are in rows with features in columns )
 * @param Y a column vector where yi corresponds to data instance xi.
 *
 */
void LR::fit( Eigen::MatrixXd& X_in, const Eigen::VectorXd& Y )
{
    if ( feature_names_.size() != X_in.cols()+1 ) {
        throw std::out_of_range( "Feature names vector and input data do not have the same length." );
    }

    double loss{std::numeric_limits<double>::max()};
    uint32_t i = 0;
    Eigen::MatrixXd I_n, I_f, pinv, temp_matrix, S, r;
    Eigen::VectorXd new_weights, y_hat, error_vector, z;

    // add the bias term.
    Eigen::MatrixXd X( X_in.rows(), X_in.cols()+1 );
    X.col(0) = Eigen::VectorXd::Ones( X.rows() );
    X.rightCols( X_in.cols() ) = X_in;

    // n x n identity matrix (elements in dataset)
    I_n = Eigen::MatrixXd::Identity( X.rows(), X.rows() );
    // f x f identity matrix (individual features including w0 term)
    I_f = Eigen::MatrixXd::Identity( X.cols(), X.cols() );

    // all zeros.
    y_hat = Eigen::VectorXd::Zero(Y.size());
    sigmoid( y_hat );
    error_vector = y_hat.array() * ( 1.0 - y_hat.array() );

    // row-wise scaling a more mathy way (maybe more computationally complex)
    S = I_n * error_vector.asDiagonal();
    r = learning_rate_ * I_f.array();

    weights_ = (X.transpose() * S * X + r).inverse() * ( X.transpose() * Y );

    while ( loss > loss_threshold_ && i < maximum_iterations_ ) { 

        // weights_ is a column vector: n x f * f x 1 = n x 1
        y_hat = X * weights_; 
        sigmoid( y_hat );

        error_vector = y_hat.array() * ( 1.0 - y_hat.array() );
        S            = I_n * error_vector.asDiagonal();
		pinv         = S.completeOrthogonalDecomposition().pseudoInverse();
        z            = X * weights_ + pinv * (Y - y_hat);
        r            = learning_rate_ * weights_.norm() * I_f;
		temp_matrix  = X.transpose() * S * X + r;
		pinv         = temp_matrix.completeOrthogonalDecomposition().pseudoInverse();
        new_weights  = pinv * X.transpose() * S * z;
        loss         = ( new_weights.array() - weights_.array() ).matrix().norm();
        weights_     = new_weights;

        ++i;
    }
    clog->trace( "Iterations: {} final loss: {}", i, loss );
}

void LR::fit( Eigen::MatrixXd&& X, const Eigen::VectorXd&& Y )
{
    fit( X, Y );
}

/**
 * Fit data matrix X and labels Y to a logistic regression model using the set learning parameters.
 *
 * This instance of LR will have its internal weights set correctly.
 *
 * @param X the data matrix ( data instances are in rows with features in columns )
 * @param Y a column vector where yi corresponds to data instance xi.
 *
 */
void LR::fit_non_irls( Eigen::MatrixXd& X, const Eigen::VectorXd& Y)
{
    if ( feature_names_.size() != X.cols()+1 ) {
        throw std::out_of_range( "Feature names vector and input data do not have the same length." );
    }

    Eigen::VectorXd y_hat = Eigen::VectorXd::Zero(Y.size());

    // add the bias term.
    X.conservativeResize( X.rows(), X.cols()+1 );
    X.col( X.cols()-1 ) = Eigen::VectorXd::Ones( X.rows() );

    weights_ = get_random_weights( X.cols() );

    double loss{std::numeric_limits<double>::max()};
    uint32_t iterations{0};

    while ( loss > loss_threshold_ && iterations < maximum_iterations_ ) { 

        clog->trace( "{} weights: {}", iterations, weights_.transpose() );

        y_hat = X * weights_;
        sigmoid( y_hat );

        clog->trace( "{} y_hat: {}", iterations, y_hat.transpose() );

        Eigen::VectorXd error = y_hat - Y;          // sorta backward but avoid the negative sign below.

        // this had a regularizer: (1-lambda)/Y.size()
        Eigen::VectorXd gradient = X.transpose() * error / Y.size() * learning_rate_;

        weights_ -= gradient;

        loss = cross_entropy_loss( Y, y_hat );

        clog->trace( "{} loss : {}", iterations, loss );

        ++iterations;
    }
}

void LR::fit_non_irls( Eigen::MatrixXd&& X, const Eigen::VectorXd&& Y)
{
    fit( X, Y );
}

// takes a row vector.
double LR::predict_sigmoid( const std::vector<double>& V_in ) const
{
    // move the vector into an Eigen row vector.
    const Eigen::Matrix<double,1,Eigen::Dynamic> V = Eigen::Map<const Eigen::Matrix<double,1,Eigen::Dynamic>>( V_in.data(), V_in.size() );
    return predict_sigmoid( V );
}

// takes a row vector.
double LR::predict_sigmoid( const Eigen::Matrix<double,1,Eigen::Dynamic>& V_in ) const
{
    // row vector. first element is the bias term (not associated with a data field).
    Eigen::Matrix<double,1,Eigen::Dynamic> V( 1, V_in.cols()+1 );
    V(0,0) = 1.0;
    V.rightCols( V_in.cols() ) = V_in;

    // compute the values for this data based on the modeled weights.
    return sigmoid( V * weights_ );
}

Eigen::VectorXd LR::predict_sigmoids( const Eigen::MatrixXd& X_in ) const
{
    Eigen::MatrixXd X( X_in.rows(), X_in.cols()+1 );
    X.col(0) = Eigen::VectorXd::Ones( X.rows() );
    X.rightCols( X_in.cols() ) = X_in;

    // compute the values for this data based on the modeled weights.
    Eigen::VectorXd y_hat = X * weights_;
    sigmoid( y_hat );
    return y_hat;
}

/**
 * Using logistics regression as a binary classifiers.
 *
 * @return 0 indicates value is less than or equal to the threshold; 1 it exceeds the threshold.
 */
int LR::classify( double pred_y ) const
{
    return static_cast<int>( pred_y > classification_threshold_ );
}

// takes a row vector.
int LR::classify( const Eigen::Matrix<double,1,Eigen::Dynamic>& V_in ) const
{
    return classify( predict_sigmoid( V_in ) );
}

// takes a row vector.
int LR::classify( const std::vector<double>& V_in ) const
{
    return classify( predict_sigmoid( V_in ));
}

Eigen::VectorXi LR::predict(Eigen::MatrixXd& X_in) const
{
    Eigen::VectorXd y_pred = predict_sigmoids( X_in );

    // apply a binary indicator function across the y_hat vector to determine classes based on threshold.
    return y_pred.unaryExpr( [this]( double x ){ return static_cast<int>( x > classification_threshold_ ); } );
}

/**
 * @return a copy of the feature names WITHOUT the "BIAS" feature.
 */
StrVector LR::get_feature_names() const
{
    if ( feature_names_.empty() ) {
        throw std::out_of_range( "Empty feature names vector." );
    }

    // remove the "BIAS" feature.
    StrVector r{ feature_names_.begin()+1, feature_names_.end() };
    return r;
}

const Eigen::VectorXd& LR::get_model_weights() const 
{
    return weights_;
}

void LR::saveWeights(const std::string& filename)
{
    //save the model (save the weight ) into filename. 
    std::ofstream ofile{ filename, std::ios::trunc };
    if (!ofile.is_open()){
        throw std::invalid_argument( "Unable to open input file: " + filename );
    }

    if ( weights_.size() != feature_names_.size() ) {
        throw std::out_of_range( "Weights vector and feature names vector do not have the same length." );
    }

    for ( std::size_t i = 0; i < weights_.size(); ++i ) {
        ofile << feature_names_[i] << ',' << weights_( i ) << '\n';
    }
}

void LR::loadWeights(const std::string& filename)
{
    //load the model (load the weight ) from filename.
    std::ifstream ifile{filename};
    if (!ifile.is_open()){
        throw std::invalid_argument( "Unable to open input file: " + filename );
    }

    // cannot immediately instantiate the correctly sized matrix, thus use a vector and transfer when size is known.
    std::vector<double> w;
    feature_names_.clear();
    std::string line;

    while ( getline( ifile, line ) ) {
        // comma delimited.
        auto parts = utilities::split( line );
        if ( parts.size() == 2 ) {

            w.push_back( std::stod( utilities::strip(parts[1]) ) );

            auto feature_name = utilities::strip( parts[0] );
            feature_names_.push_back( feature_name );
        } else {
            throw std::invalid_argument("Logistic regression weights file has incorrect format; weights vector NOT set.");
        }
    }

    //initialize VectorXd with std::vector
    weights_ = Eigen::VectorXd::Map( w.data(), w.size() );
}


