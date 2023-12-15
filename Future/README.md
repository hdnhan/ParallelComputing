## fastest
If you launch several computational tasks in parallel, and you only get some fastest reults.
- `std::async` + `std::future` => The rest of the tasks still continue and the main thread will wait for them to exit or do other things.
- `std::thread` + `std::promise` + `std::future` => The rest of the tasks will get ignored, the main thread doesn't need to wait for them, and can either exit or do other things. `std::thread` can either `join` or `detach`.
- `std::atomic` can be used to cancel a task.

## Results
```
Estimate pi by integral: pi = 4 * int_0^1 dx / (1 + x^2)
Normal  : 2.66642s (2666.42ms)
Parallel: 0.846538s (846.538ms)
diff(pi1, pi2): 4.05898e-13

Calculate dot product of two vectors
Normal  : 0.204422s (204.422ms)
Parallel: 0.061346s (61.346ms)
Parallel: 0.062482s (62.482ms)
Parallel: 0.060567s (60.567ms)
diff(res1, res2): 2.48104e-06
diff(res1, res3): 2.48104e-06
diff(res1, res4): 2.48104e-06

Get six fastest tasks, four other tasks will be ignored
19
48
47
49
50
14
44
9
32
40
End task 2 in 9s
End task 2 in 14s
End task 2 in 19s
End task 2 in 32s
End task 2 in 40s
End task 2 in 44s
Finished tasks: 9 14 19 32 40 44 
4 tasks...
26
26
3
38
11
32
48
13
27
38
End task 2 in 47s
End task 1 in 3s
End task 2 in 48s
End task 2 in 49s
End task 2 in 50s
End task 1 in 11s
End task 1 in 13s
End task 1 in 26s
End task 1 in 26s
End task 1 in 27s
Finished tasks: 3 11 13 26 26 27 
4 tasks...
End task 1 in 32s
End task 1 in 38s
End task 1 in 38s
End task 1 in 48s
Exit main!
```