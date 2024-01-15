#!/bin/bash

# create a for to create files path
for i in {0..30}
do
    # create input and output path for each image
    input_path="input_images/a${i}.pgm"
    output_path1="output_images/a${i}-1.pgm"
    output_path2="output_images/a${i}-2.pgm"

    # Run the image processing program by passing the arguments.
    ./p $input_path $output_path1 $output_path2
done