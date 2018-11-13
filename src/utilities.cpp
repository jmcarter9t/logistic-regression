#include "utilities.hpp"
#include <iostream>
#include <cmath>

const std::string utilities::DELIMITERS = " \f\n\r\t\v";

std::size_t UInt64PairHash::operator()( const UInt64Pair& p ) const 
{
    return p.first * 31 + p.second;
}

StrVector utilities::split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}
    
std::string& utilities::rstrip( std::string& s )
{
  return s.erase( s.find_last_not_of( utilities::DELIMITERS ) + 1 );
}

std::string& utilities::lstrip( std::string& s )
{
  return s.erase( 0, s.find_first_not_of( utilities::DELIMITERS ) );
}

std::string& utilities::strip( std::string& s )
{
  return utilities::lstrip( rstrip ( s ));
}

StrPair utilities::split_attribute( const std::string& s, char delim) {
    StrPair r;
    size_t pos = s.find(delim);
    if (pos < std::string::npos) {
        r.first = s.substr(0,pos);
        pos += 1;
        if (pos < std::string::npos) {
            r.second = s.substr(pos,s.size());
        }
    } 
    return r;
}

bool utilities::are_equal(double a, double b, double epsilon) {
    return std::fabs(a - b) < epsilon;
}

bool utilities::directory_exists( const std::string& path )
{
    struct stat info;

    if ( stat( path.c_str(), &info ) != 0 ) {
        return false;
    } 
    
    return ( info.st_mode & S_IFDIR );
}

bool utilities::make_directory( const std::string& path, mode_t mode )
{
    struct stat info;
    int status = 0;         // success.

    if ( stat( path.c_str(), &info ) != 0 ) {
        // file doesn't exist
        if ( mkdir( path.c_str(), mode ) != 0 && errno != EEXIST ) {
            // error returned that was NOT file already exists
            status = -1;
        }
    }  else if ( !S_ISDIR( info.st_mode ) ) {
        // exists and is a directory.
        errno = ENOTDIR;
        status = -1;
    }
    
    return ( status );
}
