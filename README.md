# Project 2: Vector Quantization and Discrete Cosine Transform
In this project, we will implement two lossy compression methods for image using vector quantization and transform coding. Write a clear detailed report of your findings. 

NOTE: Submit your program source code on GitHub Classroom, and a project report on Gradescope.

## Part 1 - Vector Quantization
Write a program to implement vector quantization on a gray-scale image using a "vector" that consists of
a 4x4 block of pixels. Design your codebook using all the blocks in the image as training data, using the Generalized Lloyd algorithm. Then quantize the image using your codebook. Explore the impact of different codebook sizes, for example, $L=128$ and $L=256$.

Next, train your codebook on a collection of 10 images, and quantize your original image using the new codebook. Compare your results on the new codebook to your previous results, and explain any differences.

command:
python3 vector_quantization.py

Note: change codebook size and image names in main().

## Part 2 - Discrete Cosine Transform 
Write a program that examines the effect of approximating an image with a partial set of DCT coefficients. Using an $8 \times 8$ DCT, reconstruct the image with $K<64$ coefficients, when $K=2, 4, 8, 16$, and $32$. How many coefficients are necessary to provide a "satisfactory" reconstruction?

Define how you characterize "satisfactory" reconstruction.

command:
python3 DCT.py

Note: change K value in main()

Python Version Information

Python 3.11.1 (v3.11.1:a7a450f84a, Dec  6 2022, 15:24:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.

**Note: For each part, you need to test your method on at least two different images.** 

A collection of images are available on the course website [HERE] (https://engineering.purdue.edu/~zhu0/ece634/sample_image.zip). 


