/**
 * @internal
 * @author Oded Green                                                  <br>
 *         Georgia Institute of Technology, Computational Science and Engineering                   <br>
 *         ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */

#include "Static/KatzCentrality/katz.cuh"


using namespace xlib;
using namespace std;
typedef int32_t length_t;


namespace hornet_alg {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in the streaming case.

katzCentrality::katzCentrality(HornetGPU& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balacing(hornet)
									    {

    deviceKatzData = register_data(hostKatzData);
	memReleased = true;
}

katzCentrality::~katzCentrality() {
	release();
}

void katzCentrality::setInitParameters(int32_t maxIteration_,int32_t K_,int32_t maxDegree_, bool isStatic_){
	hostKatzData.K=K_;
	hostKatzData.maxDegree=maxDegree_;
	hostKatzData.alpha = 1.0/((double)hostKatzData.maxDegree+1.0);

	hostKatzData.maxIteration=maxIteration_;
	isStatic = isStatic_;

	if(maxIteration_==0){
		cout << "Number of max iterations should be greater than zero" << endl;
		return;
	}
}


void katzCentrality::init(){

	if(memReleased==false){
		release();
		memReleased=true;
	}
	hostKatzData.nv = hornet.nV();

	if(isStatic==true){
		gpu::allocate(hostKatzData.nPathsData, hostKatzData.nv*2);
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		gpu::allocate(hostKatzData.nPathsData, hostKatzData.nv*hostKatzData.maxIteration);
		gpu::allocate(hostKatzData.nPaths, hostKatzData.maxIteration);
		hPathsPtr = (ulong_t**)malloc(hostKatzData.maxIteration* sizeof(ulong_t*));

		for(int i=0; i< hostKatzData.maxIteration; i++){
			hPathsPtr[i] = (hostKatzData.nPathsData+(hostKatzData.nv)*i);
		}
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];
		gpu::copyHostToDevice(hPathsPtr,hostKatzData.maxIteration,hostKatzData.nPaths);
	}

	gpu::allocate(hostKatzData.KC, hostKatzData.nv);
	gpu::allocate(hostKatzData.lowerBound, hostKatzData.nv);
	gpu::allocate(hostKatzData.upperBound, hostKatzData.nv);

	gpu::allocate(hostKatzData.isActive, hostKatzData.nv);
	gpu::allocate(hostKatzData.vertexArraySorted, hostKatzData.nv);
	gpu::allocate(hostKatzData.vertexArrayUnsorted, hostKatzData.nv);
	gpu::allocate(hostKatzData.lowerBoundSorted, hostKatzData.nv);
	gpu::allocate(hostKatzData.lowerBoundUnsorted, hostKatzData.nv);

	syncDeviceWithHost();
	reset();
}

void katzCentrality::reset(){
	hostKatzData.iteration = 1;

	if(isStatic==true){
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];
	}
	syncDeviceWithHost();
}

void katzCentrality::release(){
	if(memReleased==true)
		return;
	memReleased=true;

	gpu::free(hostKatzData.nPathsData);

	if (!isStatic){
		gpu::free(hostKatzData.nPaths);
		// freeHostArray(hPathsPtr);
		free(hPathsPtr);
	}

	gpu::free(hostKatzData.KC);
	gpu::free(hostKatzData.lowerBound);
	gpu::free(hostKatzData.upperBound);

	gpu::free(hostKatzData.vertexArraySorted);
	gpu::free(hostKatzData.vertexArrayUnsorted);
	gpu::free(hostKatzData.lowerBoundSorted);
	gpu::free(hostKatzData.lowerBoundUnsorted);

}

void katzCentrality::run(){
	forAllVertices<katz_operators::init>(hornet,deviceKatzData);
	hostKatzData.iteration = 1;

	hostKatzData.nActive = hostKatzData.nv;

	while(hostKatzData.nActive> hostKatzData.K && hostKatzData.iteration < hostKatzData.maxIteration){

		hostKatzData.alphaI          = pow(hostKatzData.alpha,hostKatzData.iteration);
		hostKatzData.lowerBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha));
		hostKatzData.upperBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha*(double)hostKatzData.maxDegree));
		hostKatzData.nActive = 0; // Each iteration the number of active vertices is set to zero.

		syncDeviceWithHost(); // Passing constants to the device.
		forAllVertices<katz_operators::initNumPathsPerIteration>(hornet,deviceKatzData);
        forAllEdges<katz_operators::updatePathCount>(hornet, deviceKatzData);
		forAllVertices<katz_operators::updateKatzAndBounds>(hornet,deviceKatzData);
		syncHostWithDevice();

		hostKatzData.iteration++;
		if(isStatic){
			// Swapping pointers.
			ulong_t* temp = hostKatzData.nPathsCurr; hostKatzData.nPathsCurr=hostKatzData.nPathsPrev; hostKatzData.nPathsPrev=temp;
		}else{
			hostKatzData.nPathsPrev = hPathsPtr[hostKatzData.iteration - 1];
			hostKatzData.nPathsCurr = hPathsPtr[hostKatzData.iteration - 0];
		}
		length_t oldActiveCount 	= hostKatzData.nActive;
		hostKatzData.nPrevActive 	= hostKatzData.nActive;
		hostKatzData.nActive = 0; // Resetting active vertices for sorting operations.
		syncDeviceWithHost();

		// Notice that the sorts the vertices in an incremental order based on the lower bounds.
		// The algorithms requires the vertices to be sorted in an decremental fashion.
		// As such, we use the nPrevActive variables to store the number of previous active vertices
		// and are able to find the K-th from last vertex (which is essentially going from the tail of the array).
		xlib::CubSortByKey<double,vid_t> sorter(hostKatzData.lowerBoundUnsorted,hostKatzData.vertexArrayUnsorted,oldActiveCount,hostKatzData.lowerBoundSorted, hostKatzData.vertexArraySorted);
		sorter.run();

		forAllVertices<katz_operators::countActive>(hornet,hostKatzData.vertexArrayUnsorted,oldActiveCount,deviceKatzData);
		syncHostWithDevice();
		// printKMostImportant();

		// cout << "Active  : " << hostKatzData.nActive << endl;
	}
	// cout << "@@ " << hostKatzData.iteration << " @@" << endl;
	syncHostWithDevice();
}
// This function should only be used directly within run() and is currently commented out due to 
// to large execution overheads.
void katzCentrality::printKMostImportant(){

		ulong_t* nPathsCurr = (ulong_t*) malloc(hostKatzData.nv* sizeof(ulong_t));
		ulong_t* nPathsPrev = (ulong_t*) malloc(hostKatzData.nv* sizeof(ulong_t));
		vid_t* vertexArray = (vid_t*) malloc(hostKatzData.nv* sizeof(vid_t));
		vid_t* vertexArrayUnsorted = (vid_t*) malloc(hostKatzData.nv* sizeof(vid_t));
		double* KC         = (double*) malloc(hostKatzData.nv* sizeof(double));
		double* lowerBound = (double*) malloc(hostKatzData.nv* sizeof(double));
		double* upperBound = (double*) malloc(hostKatzData.nv* sizeof(double));

		gpu::copyDeviceToHost(hostKatzData.lowerBound,hostKatzData.nv,lowerBound);
		gpu::copyDeviceToHost(hostKatzData.upperBound,hostKatzData.nv, upperBound);
		gpu::copyDeviceToHost(hostKatzData.KC,hostKatzData.nv,KC);
		gpu::copyDeviceToHost(hostKatzData.vertexArraySorted,hostKatzData.nv,vertexArray);
		gpu::copyDeviceToHost(hostKatzData.vertexArrayUnsorted,hostKatzData.nv,vertexArrayUnsorted);


		if(hostKatzData.nPrevActive>hostKatzData.K)
			for (int i=hostKatzData.nPrevActive-1; i>=(hostKatzData.nPrevActive-hostKatzData.K); i--){
				vid_t j=vertexArray[i];
				printf("%d\t\t %e\t\t %e\t\t %e\t\t %e\t\t \n",j,KC[j],upperBound[j],lowerBound[j],upperBound[j]-lowerBound[j]);
			}

		free(nPathsCurr);
		free(nPathsPrev);
		free(vertexArray);
		free(vertexArrayUnsorted);
		free(KC);
		free(lowerBound);
		free(upperBound);
		
}

length_t katzCentrality::getIterationCount(){
	syncHostWithDevice();
	return hostKatzData.iteration;
}


}// hornetAlgs namespace
