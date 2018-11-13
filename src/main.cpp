#include <iostream>
#include <fstream>
#include "lr.hpp"
#include "confusion_matrix.hpp"

bool load_data( const std::string& filename, StrVector& feature_names, Eigen::MatrixXd& D )
{
    //load the model (load the weight ) from filename.
    std::ifstream ifile{filename};
    if (!ifile.is_open()){
        throw std::invalid_argument( "Unable to open input file: " + filename );
    }

    std::string line;
    std::vector<double> d;
    uint32_t elements{0};
    uint32_t rows{0};
    uint32_t cols{0};

    bool header = true;

    while ( getline( ifile, line ) ) {

        StrVector parts = utilities::split( utilities::strip(line) );

        if ( !header ) {
            elements += parts.size();
            if ( parts.size() > cols ) {
                cols = parts.size();
            }

            for ( auto& p : parts ) {
                d.push_back( std::stod( p ) );
            }
            ++rows;
        } else {
            feature_names = parts;
            feature_names.pop_back();       // remove the class label.
            header = false;
        }
    }

    if ( ( feature_names.size() != cols-1 ) || elements != rows * cols ) {
        throw std::invalid_argument( "malformation of data in file." );
    }

    // ensure this is row-major
    D = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>( d.data(), rows, cols );
    return true;
}

int main( int argc, char* argv[] ){

    auto clog = spdlog::stdout_color_mt("clog");
    clog->set_level(spdlog::level::trace);
    clog->set_pattern("%v");

    if ( utilities::make_directory( "logs", S_IRWXU | S_IRWXG | S_IROTH  ) != 0 ) {
        clog->critical("Failed to make non-existant logs directory.");
        return EXIT_FAILURE;
    }

    auto elog = spdlog::basic_logger_mt( "elog", "logs/lr.log", true );         // truncate each time.
    StrVector args{ argv, argv+argc };

    if ( args.size() < 2 ) {
        clog->error("Usage:\t ./lr <train|test> .... <data> <output file>");
        return EXIT_FAILURE;
    }

    int action = 3;
    if ( args[1] == "train" ) action = 1;
    else if ( args[1] == "test" ) action = 2;

    try {

        switch ( action ) {

            case 1:  // train
                {
                    if ( argc != 7 ) {
                        clog->error("Usage:\t ./lr <train> <iterations> <loss_threshold> <learning_rate> <data> <output file>");
                        clog->error("\t ./lr <test> <weights> <class threshold> <data>");
                        return EXIT_FAILURE;
                    }

                    uint32_t iterations = std::stoi( argv[2] );
                    double loss_threshold = std::stod( argv[3] );
                    double learning_rate = std::stod( argv[4] );

                    Eigen::MatrixXd D;
                    StrVector feature_names;
                    load_data( argv[5], feature_names, D );

                    // Eigen::MatrixXd D       = load_data( argv[6] );
                    Eigen::MatrixXd X_train = D.leftCols( D.cols()-1 );
                    Eigen::VectorXd y_train = D.col( D.cols()-1 );

                    clog->info( "Input File: {}", argv[5] );
                    clog->info( "Iterations: {}", iterations );
                    clog->info( "Loss Threshold: {}", loss_threshold );
                    clog->info( "Learning Rate: {}", learning_rate );

                    LR engine{ feature_names, iterations, loss_threshold, learning_rate };
                    engine.fit( X_train, y_train );
                    engine.saveWeights( argv[6] );
                }
                break;

            case 2:
                {

                    if ( argc != 5 ) {
                        clog->error("Usage:\t ./lr <train> <iterations> <loss_threshold> <learning_rate> <data> <output file>");
                        clog->error("\t ./lr <test> <weights> <class_threshold> <data>");
                        return EXIT_FAILURE;
                    }

                    LR classify_engine{ argv[2], std::stod( argv[3] ) };
                    //load the model (load the weight ) from filename.
                    std::ifstream ifile{ argv[4] };
                    if (!ifile.is_open()){
                        clog->error( "Unable to open input file: {}", argv[4] );
                        return EXIT_FAILURE;
                    }

                    std::string line;
                    std::vector<int> y_pred;
                    std::vector<int> y_test;

                    bool header = true;

                    // large file processing - line by line.
                    while ( getline( ifile, line ) ) {

                        if ( header ) {
                            header = false;
                            continue;
                        }

                        StrVector parts = utilities::split( utilities::strip(line) );
                        std::vector<double> vp;
                        // skip the first two fields in the testing data -- this is what the python script generates from the CVPM tool.
                        // std::transform( parts.begin()+2, 
                        std::transform( parts.begin(), 
                                parts.end(),
                                std::back_inserter( vp ), []( std::string& s ) -> double { return std::stod( s ); } );
                        // last piece is the actual class of the instance.
                        y_test.push_back( static_cast<int>( vp.back() ) );
                        // get rid of it for classification.
                        vp.pop_back();

                        y_pred.push_back( classify_engine.classify( vp ) );
                    }

                    Confusion_Matrix cm{ y_test, y_pred };
                    clog->info( "{}", cm );
                }
                break;
            default:
                clog->error("Usage:\t ./lr <train> <iterations> <loss_threshold> <learning_rate> <data>");
                clog->error("\t ./lr <test> <weights> <class_threshold> <data>");
                return EXIT_FAILURE;
                break;
        }
    } catch (std::exception& e ) {
        clog->error("{}", e.what() );
        return EXIT_FAILURE;
    }

    return 0;
}
