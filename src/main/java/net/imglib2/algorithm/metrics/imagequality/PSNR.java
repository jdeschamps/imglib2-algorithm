package net.imglib2.algorithm.metrics.imagequality;

import java.util.Arrays;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

/**
 * Compute the peak signal-to-noise ratio (PSNR) between a reference and a processed image. The
 * metrics runs on the whole image, whether 2D or 3D. In order to get individual slice PSNR, run
 * the metrics on each slice independently.
 *
 * @author Joran Deschamps
 * @see <a href="https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio">PSNR on Wikipedia</a>
 */
public class PSNR
{
	public static < T extends RealType< T > > double computeMetrics( final RandomAccessibleInterval< T > reference, final RandomAccessibleInterval< T > processed )
	{
		if ( !Arrays.equals( reference.dimensionsAsLongArray(), processed.dimensionsAsLongArray() ) )
			throw new IllegalArgumentException( "Image dimensions must match." );

		long nPixels = Arrays.stream( reference.dimensionsAsLongArray() ).reduce( 1, ( a, b ) -> a * b );

		// convert to float
		final RandomAccessibleInterval< DoubleType > refAsDouble = Converters.convert( reference, ( i, o ) -> o.set( i.getRealDouble() ), new DoubleType() );
		final RandomAccessibleInterval< DoubleType > procAsDouble = Converters.convert( processed, ( i, o ) -> o.set( i.getRealDouble() ), new DoubleType() );

		double mse = 0.;
		double max = Double.NEGATIVE_INFINITY;
		final Cursor< DoubleType > cu = Views.iterable( refAsDouble ).localizingCursor();
		final RandomAccess< DoubleType > ra = procAsDouble.randomAccess();
		while ( cu.hasNext() )
		{
			double dRef = cu.next().getRealDouble();

			if ( dRef > max )
				max = dRef;

			ra.setPosition( cu );
			double dProc = ra.get().getRealDouble();

			mse += ( dRef - dProc ) * ( dRef - dProc ) / nPixels;
		}

		return mse > 0 ? 20 * Math.log10( max / Math.sqrt( mse ) ) : Double.NaN;
	}
}
