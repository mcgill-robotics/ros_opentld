/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * NNClassifier.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#ifndef NNCLASSIFIER_H_
#define NNCLASSIFIER_H_

#include <vector>

#include <opencv/cv.h>

#include "NormalizedPatch.h"
#include "DetectionResult.h"

namespace tld
{

class NNet
{
public:
  NNet(float x,float y, float z) : epsilon_(x), threshold(y), object_type(z) {}
  float getX() const {return epsilon_;}
  float getY() const {return threshold;}
  float getZ() const {return objecttype;}
  void setX(float val) {epsilon_ = val;}
  void setY(float val) {threshold = val;}
  void setZ(float val) {objecttype = val;}
private:
  float epsilon_;
  float threshold;
  float objecttype;
};

class NNClassifier
{
    float ncc(float *f1, float *f2);
public:
    bool enabled;
    bool threshold_NNet_loaded = false;

    int *windows;
    float thetaFP;
    float thetaTP;
    DetectionResult *detectionResult;
    std::vector<NormalizedPatch>* falsePositives;
    std::vector<NormalizedPatch>* truePositives;

    NNClassifier();
    virtual ~NNClassifier();

    void release();
    float classifyPatch(NormalizedPatch *patch);
    float classifyBB(const cv::Mat &img, cv::Rect *bb);
    float classifyWindow(const cv::Mat &img, int windowIdx);
    void learn(std::vector<NormalizedPatch> patches);
    bool filter(const cv::Mat &img, int windowIdx);
};

} /* namespace tld */
#endif /* NNCLASSIFIER_H_ */
