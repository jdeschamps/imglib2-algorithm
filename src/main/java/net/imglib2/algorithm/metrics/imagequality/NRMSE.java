package net.imglib2.algorithm.metrics.imagequality;

import java.util.Arrays;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.stats.ComputeMinMax;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/**
 * Compute the normalised root mean squared error (NRMSE) between a reference and a processed image.
 * The metrics runs on the whole image, whether 2D or 3D. In order to get individual slice NRMSE, run
 * the metrics on each slice independently.
 * <p>
 * Three normalization methods are available:
 * - euclidean: NRMSE = sqrt(MSE(reference, processed)) * sqrt(N) / || reference ||
 * - min-max: NRMSE = sqrt(MSE(reference, processed)) * sqrt(N) / |max(reference)-min(reference)|
 * - mean: NRMSE = sqrt(MSE(reference, processed)) * sqrt(N) / mean(reference)
 *
 * @author Joran Deschamps
 */
public class NRMSE
{
	/**
	 * Normalization methods.
	 */
	public enum Normalization
	{
		EUCLIDEAN,
		MINMAX,
		MEAN
	}

	/**
	 * Compute the normalised root mean squared error (NRMSE) score between reference and processed images. The metrics
	 * run on the whole image (regardless of the dimensions).
	 * <p>
	 * Three normalization methods are available:
	 * - euclidean: NRMSE = sqrt(MSE(reference, processed)) * sqrt(N) / || reference ||
	 * - min-max: NRMSE = sqrt(MSE(reference, processed)) / |max(reference)-min(reference)|
	 * - mean: NRMSE = sqrt(MSE(reference, processed)) / mean(reference)
	 *
	 * @param reference
	 * 		Reference image
	 * @param processed
	 * 		Processed image
	 * @param <T>
	 * 		Type of the image pixels
	 *
	 * @return Metrics score
	 */
	public static < T extends RealType< T > > double computeMetrics(
			final RandomAccessibleInterval< T > reference,
			final RandomAccessibleInterval< T > processed,
			final Normalization norm )
	{
		if ( !Arrays.equals( reference.dimensionsAsLongArray(), processed.dimensionsAsLongArray() ) )
			throw new IllegalArgumentException( "Image dimensions must match." );

		// get mse
		double mse = MSE.computeMetrics( reference, processed );

		double nFactor = 0;
		if ( Normalization.EUCLIDEAN.equals( norm ) )
		{
			nFactor = getEuclideanNorm( reference );
		}
		else if ( Normalization.MINMAX.equals( norm ) )
		{
			nFactor = getMinMaxNorm( reference );
		}
		else
		{
			nFactor = getMean( reference );
		}

		return nFactor > 0 ? Math.sqrt( mse ) / nFactor : Double.NaN;
	}

	protected static < T extends RealType< T > > double getEuclideanNorm(
			final RandomAccessibleInterval< T > reference )
	{
		// get image size
		final long nPixels = Arrays.stream( reference.dimensionsAsLongArray() ).reduce( 1, ( a, b ) -> a * b );

		if ( nPixels > 0 )
		{
			double ms = 0.;
			final Cursor< T > cu = Views.iterable( reference ).localizingCursor();
			while ( cu.hasNext() )
			{
				double dRef = cu.next().getRealDouble();
				ms += dRef * dRef / nPixels; // division here to avoid precision loss due to overflow
			}

			return Math.sqrt( ms );
		}
		return Double.NaN;
	}

	protected static < T extends RealType< T > > double getMinMaxNorm(
			final RandomAccessibleInterval< T > reference )
	{
		T min = reference.randomAccess().get().copy();
		T max = min.copy();

		ComputeMinMax.computeMinMax( reference, min, max );

		return max.getRealDouble() - min.getRealDouble();
	}

	protected static < T extends RealType< T > > double getMean(
			final RandomAccessibleInterval< T > reference )
	{
		// get image size
		final long nPixels = Arrays.stream( reference.dimensionsAsLongArray() ).reduce( 1, ( a, b ) -> a * b );

		if ( nPixels > 0 )
		{
			double mean = 0.;
			final Cursor< T > cu = Views.iterable( reference ).localizingCursor();
			while ( cu.hasNext() )
			{
				double dRef = cu.next().getRealDouble();
				mean += dRef / nPixels; // division here to avoid precision loss due to overflow
			}
			return mean;
		}

		return Double.NaN;
	}
}