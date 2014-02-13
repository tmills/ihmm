-------------------------------------------------------
iHMM Sampling Library
-------------------------------------------------------

This package contains code for learning and sampling form finite HMM's and 
infinite HMM's. The code is dependent on the lightspeed and fastfit 
(http://research.microsoft.com/~minka/software/lightspeed/, 
http://research.microsoft.com/~minka/software/fastfit/) packages by Tom 
Minka. These libraries must be on the Matlab path in order for the sampling
algorithms to work.



Some notes on the iHMM's with multinomial output:
* TestiHmmGibbsSampler.m runs the Gibbs sampler on an iHMM with multinomial
output. It should clarify how to use iHmmSampleGibbs.m; also use the
command "help iHmmSampleGibbs" for more information on the parameters.
* TestiHmmBeamSampler.m runs the beam sampler on an iHMM with multinomial
output. It should clarify how to use iHmmSampleBeam.m; also use the
command "help iHmmSampleBeam" for more information on the parameters.
* The joint log likelihood is the following: p(s,y|beta,alpha,gamma,H).


Some notes on the iHMM's with normal output:
* TestiHmmNormalGibbsSampler.m runs the Gibbs sampler on an iHMM with 
normal output. It should clarify how to use iHmmNormalSampleGibbs.m; also 
use the command "help iHmmNormalSampleGibbs" for more information on the 
parameters.
* TestiHmmNormalBeamSampler.m runs the beam sampler on an iHMM with 
normal output. It should clarify how to use iHmmNormalSampleBeam.m; also 
use the command "help iHmmNormalSampleBeam" for more information on the 
parameters.
* The joint log likelihood is the following: p(s,y|beta,alpha,gamma,H).
* Currently only the means of the normal distribution are sampled.


Change Log
----------

v0.5 - 21/07/2010
* Removed dependency from lightspeed (now using statistics toolbox).
* Updated to newer matlab versions.

v0.4 - 24/8/2009
* Removed the need for Stirling numbers.
* Fixed a bug in the backtwards sampling stage.

v 0.3 - 6/02/2009
* Vectorized some of the operations in the beam samplers.

v 0.2 - 6/08/2008
* Added the beam sampler for the iHMM.

v 0.1 - 10/02/2008
* Original release of the Gibbs sampler for the iHMM.



===========================================================================
(C) Copyright 2008-2009, Jurgen Van Gael (jv279 -at- cam .dot. ac .dot. uk)

http://mlg.eng.cam.ac.uk/jurgen/

Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.

These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.