# NestedBuffer
This code explains the concept of nested buffer using SYCL (DPC++) and there mapping in memory.

# Code Description
-> The struct C contains a data variable and struct B also contains a data variable with a pointer to the struct C.

-> struct BBuff contains sycl buffer of both B and C 

-> struct BView contains sycl accessor to the buffer of B and C with RequireForHandler() method

-> main() contains the usual structure of sycl code with buffer, accessor, handler, and single_task Kernel which is assigning the values to those structures. Finally printing the values of the data.
