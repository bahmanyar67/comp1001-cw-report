/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//In Linux compile using : gcc image_processing.c   -o p -O3  -lm

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>


//function declarations
void Gaussian_Blur ();
void Sobel ();
void Sobel_sse2 ();
int initialize_kernel ();
void read_image (char *filename);
void read_image_and_put_zeros_around (char *filename);
void write_image2 (char *filename, unsigned char *output_image);
void openfile (char *filename, FILE ** finput);
int getint (FILE * fp);

#define SOURCE_DIR "input_images"
#define DEST_DIR "output_images"

//IMAGE DIMENSIONS
int M;
int N;


//CRITICAL POINT:these arrays are defined statically. Consider creating these arrays dynamically instead.
unsigned char *frame1;		//input image
unsigned char *filt;		//output filtered image
unsigned char *gradient;	//output image


const signed char Mask[5][5] = {	//2d gaussian mask with integers
  {2, 4, 5, 4, 2},
  {4, 9, 12, 9, 4},
  {5, 12, 15, 12, 5},
  {4, 9, 12, 9, 4},
  {2, 4, 5, 4, 2}
};

const signed char GxMask[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};

const signed char GyMask[3][3] = {
  {-1, -2, -1},
  {0, 0, 0},
  {1, 2, 1}
};

char header[100];


int main ()
{

char *in, *out, *out2;

  int i;

  // Allocate memory for in, out, and out2
    in = malloc(256);
    out = malloc(256);
    out2 = malloc(256);

  for (i = 0; i <= 30; i++)
    {

    //convert i to string in c

        char * filename = malloc(20);
        sprintf(filename, "a%d", i);


      // input_images + / + a0
      sprintf (in, "%s/%s", SOURCE_DIR, filename);
      sprintf (out, "%s/%s_blurred.pgm", DEST_DIR, filename);
      sprintf (out2, "%s/%s_edge.pgm", DEST_DIR, filename);

      // input_images/a0 + .pgm
      strcat (in, ".pgm");
      strcat (out, "_blurred.pgm");
      strcat (out2, "_edge.pgm");


      read_image (in);
      Gaussian_Blur ();
      //Sobel ();
        //Sobel_avx2 ();
        Sobel_sse2();
      write_image2 (out, filt);
      write_image2 (out2, gradient);

        free (frame1);
        free (filt);
        free (gradient);
    }

  free(in);
  free(out);
  free(out2);

  return 0;
}

void Gaussian_Blur ()
{

  int row, col, rowOffset, colOffset;
  int newPixel;
  unsigned char pix;
  //const unsigned short int size=filter_size/2;
  const unsigned short int size = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
  for (row = 0; row < N; row++)
    {
      for (col = 0; col < M; col++)
	{
	  newPixel = 0;
	  for (rowOffset = -size; rowOffset <= size; rowOffset++)
	    {
	      for (colOffset = -size; colOffset <= size; colOffset++)
		{

		  if ((row + rowOffset < 0) || (row + rowOffset >= N)
		      || (col + colOffset < 0) || (col + colOffset >= M))
		    pix = 0;
		  else
		    pix = frame1[M * (row + rowOffset) + col + colOffset];

		  newPixel += pix * Mask[size + rowOffset][size + colOffset];

		}
	    }
	  filt[M * row + col] = (unsigned char) (newPixel / 159);

	}
    }

}


void Sobel ()
{

  int row, col, rowOffset, colOffset;
  int Gx, Gy;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
  for (row = 1; row < N - 1; row++)
    {
      for (col = 1; col < M - 1; col++)
	{

	  Gx = 0;
	  Gy = 0;

	  /* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
	  for (rowOffset = -1; rowOffset <= 1; rowOffset++)
	    {
	      for (colOffset = -1; colOffset <= 1; colOffset++)
		{

		  Gx += filt[M * (row + rowOffset) + col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
		  Gy += filt[M * (row + rowOffset) + col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
		}
	    }

	  gradient[M * row + col] = (unsigned char) sqrt (Gx * Gx + Gy * Gy);	/* Calculate gradient strength               */
	  //gradient[row][col] = abs(Gx) + abs(Gy); // this is an optimized version of the above

	}
    }


}


void Sobel_sse2() {
    int row, col;
    __m128i Gx, Gy, result;

    const __m128i GxMask = _mm_set_epi16(-1, 0, 1, -2, 0, 2, -1, 0);
    const __m128i GyMask = _mm_set_epi16(-1, -2, -1, 0, 0, 0, 1, 2);

    for (row = 1; row < N - 1; row++) {
        for (col = 1; col < M - 1; col++) {
            __m128i neighborhood = _mm_loadu_si128((__m128i*)&filt[M * (row - 1) + col - 1]);

            Gx = _mm_mullo_epi16(neighborhood, GxMask);
            Gy = _mm_mullo_epi16(neighborhood, GyMask);

            Gx = _mm_hadd_epi16(Gx, Gx);
            Gy = _mm_hadd_epi16(Gy, Gy);

            result = _mm_add_epi16(Gx, Gy);
            result = _mm_abs_epi16(result);

            _mm_storeu_si128((__m128i*)&gradient[M * row + col], result);
        }
    }
}


void read_image (char *filename)
{

  int c;
  FILE *finput;
  int i, j, temp;

  printf ("\nReading %s image from disk ...", filename);
  finput = NULL;
  openfile (filename, &finput);


  // Allocate memory for the arrays
  frame1 = (unsigned char *) malloc (M * N * sizeof (unsigned char));
  filt = (unsigned char *) malloc (M * N * sizeof (unsigned char));
  gradient = (unsigned char *) malloc (M * N * sizeof (unsigned char));


  if ((header[0] == 'P') && (header[1] == '2'))
    {
      for (j = 0; j < N; j++)
	{
	  for (i = 0; i < M; i++)
	    {

	      if (fscanf (finput, "%d", &temp) == EOF)
		exit (EXIT_FAILURE);

	      frame1[M * j + i] = (unsigned char) temp;
	    }
	}
    }
  else if ((header[0] == 'P') && (header[1] == '5'))
    {
      for (j = 0; j < N; j++)
	{
	  for (i = 0; i < M; i++)
	    {
	      c = getc (finput);
	      frame1[M * j + i] = (unsigned char) c;
	    }
	}
    }
  else
    {
      printf ("\n problem with reading image");
      exit (EXIT_FAILURE);
    }


  fclose (finput);
  printf ("\nimage successfully read from disc\n");

}


void write_image2 (char *filename, unsigned char *output_image)
{

  FILE *foutput;
  int i, j;



  printf ("  Writing result to disk ...\n");

  if ((foutput = fopen (filename, "wb")) == NULL)
    {
      fprintf (stderr, "Unable to open file %s for writing\n", filename);
      exit (-1);
    }

  fprintf (foutput, "P2\n");
  fprintf (foutput, "%d %d\n", M, N);
  fprintf (foutput, "%d\n", 255);

  for (j = 0; j < N; ++j)
    {
      for (i = 0; i < M; ++i)
	{
	  fprintf (foutput, "%3d ", output_image[M * j + i]);
	  if (i % 32 == 31)
	    fprintf (foutput, "\n");
	}
      if (M % 32 != 0)
	fprintf (foutput, "\n");
    }
  fclose (foutput);


}


void openfile (char *filename, FILE ** finput)
{
  int x0, y0, x;

//int aa;

  if ((*finput = fopen (filename, "rb")) == NULL)
    {
      fprintf (stderr, "Unable to open file %s for reading\n", filename);
      exit (-1);
    }

  if (fscanf (*finput, "%s", header) == EOF)
    exit (EXIT_FAILURE);

  x0 = getint (*finput);	//this is M
  y0 = getint (*finput);	//this is N
  printf ("\t header is %s, while x=%d,y=%d", header, x0, y0);


  //CRITICAL POINT: AT THIS POINT YOU CAN ASSIGN x0,y0 to M,N
  // printf("\n Image dimensions are M=%d,N=%d",M,N);
  M = x0;
  N = y0;


  x = getint (*finput);		/* read and throw away the range info */
  //printf("\n range info is %d",x);

}



//CRITICAL POINT: you can define your routines here that create the arrays dynamically; now, the arrays are defined statically.



int getint (FILE * fp)		/* adapted from "xv" source code */
{
  int c, i, firstchar;		//, garbage;

  /* note:  if it sees a '#' character, all characters from there to end of
     line are appended to the comment string */

  /* skip forward to start of next number */
  c = getc (fp);
  while (1)
    {
      /* eat comments */
      if (c == '#')
	{
	  /* if we're at a comment, read to end of line */
	  char cmt[256], *sp;

	  sp = cmt;
	  firstchar = 1;
	  while (1)
	    {
	      c = getc (fp);
	      if (firstchar && c == ' ')
		firstchar = 0;	/* lop off 1 sp after # */
	      else
		{
		  if (c == '\n' || c == EOF)
		    break;
		  if ((sp - cmt) < 250)
		    *sp++ = c;
		}
	    }
	  *sp++ = '\n';
	  *sp = '\0';
	}

      if (c == EOF)
	return 0;
      if (c >= '0' && c <= '9')
	break;			/* we've found what we were looking for */

      /* see if we are getting garbage (non-whitespace) */
      // if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',')
      //  garbage=1;

      c = getc (fp);
    }

  /* we're at the start of a number, continue until we hit a non-number */
  i = 0;
  while (1)
    {
      i = (i * 10) + (c - '0');
      c = getc (fp);
      if (c == EOF)
	return i;
      if (c < '0' || c > '9')
	break;
    }
  return i;
}