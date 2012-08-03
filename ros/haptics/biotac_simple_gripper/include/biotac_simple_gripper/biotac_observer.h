/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2012, University of Pennsylvania
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of University of Pennsylvania nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* Author: Vivian Chu (chuv@grasp.upenn.edu)
*         Ian McMahon (imcmahon@grasp.upenn.edu)
*********************************************************************/


#ifndef _BIOTAC_OBSERVER
#define _BIOTAC_OBSERVER

#include <biotac_sensors/BioTacHand.h>
#include <vector>
#include <boost/thread/shared_mutex.hpp>

//================================================================
// BioTacObserver Class Defintion
//================================================================
class biotacObserver{

  private:

    // Constants
    static const int NumberFingers = 2;                      // Expected # of BioTacs 
    static const unsigned int NumberNormalizeValues = 10;    // Number of initial readings used to find the mean
   
    // Variables 
    std::vector <std::vector <int> > init_pressure_vect_;    // Store initial readings to normalize the BioTac sensors
    int pressure_mean_[NumberFingers];                       // Store the mean of the pressure
    int pressure_[NumberFingers];                            // Latest raw pressure reading from BioTac callback
   // boost::shared_mutex biotac_mutex_;                     // Lock for reading/writing to final pressure

    // Internal Functions
    void normalize_pressure();                    // Normalize BioTac Readings
    void calculate_mean();                        // Calculates the mean for PDC

  public:

    // Variables 
    static const int Left = 0;
    static const int Right = 1;
    int pressure_normalized_[NumberFingers]; 
    bool init_flag_;                     // Checks if initialization (normalization, etc) is complete
    bool renorm_flag_;                            // Checks for renormalization
    boost::shared_mutex biotac_mutex_;            // Lock for reading/writing to     final pressure

    
    // Constructor/Destructor
    biotacObserver();                             // Constructor 
    ~biotacObserver();                            // Destructor
    // Functions
    void renormalize();                           // Asks the biotacs to renormalize PDC 
    void bioTacCB(const biotac_sensors::BioTacHand::ConstPtr& msg);                              // Callback for BioTac Sensor
};
#endif // _BIOTAC_OBSERVER    
