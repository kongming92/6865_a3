Name: Charles Liu
MIT Email: cliu2014@mit.edu

How long did the assignment take?
About 6 hours

Potential issues with your solution and explanation of partial completion (for partial credit)
I believe everything is correct, although the bilateralYUV takes quite a long  time.

Any extra credit you may have implemented
None :(

Collaboration acknowledgement (but again, you must write your own code)
none

What was most unclear/difficult?
The Bilateral filtering on the YUV channels. I originally tried to rewrite the loop to simultaneously do the Y and UV channels (which is faster) but ultimately settled on using the existing bilateral code twice (Y and UV) and combining the resulting images.

What was most exciting?
Seeing the bilateral filtering finally work as anticipated and seeing the good results.

Timings:
{A1: 6.78734993935 seconds} {A2: 77.0119440556 seconds}