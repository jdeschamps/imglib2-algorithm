package net.imglib2.algorithm.metrics.segmentation;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.view.Views;
import org.junit.Test;

import static net.imglib2.algorithm.metrics.segmentation.SEGMetricsTest.getSEGBetweenRectangles;
import static net.imglib2.algorithm.metrics.segmentation.SegmentationMetricsTest.exampleIndexArray;
import static net.imglib2.algorithm.metrics.segmentation.SegmentationMetricsTest.exampleIndexArrayDims;
import static net.imglib2.algorithm.metrics.segmentation.SegmentationMetricsTest.exampleIntersectingLabels;
import static net.imglib2.algorithm.metrics.segmentation.SegmentationMetricsTest.exampleNonIntersectingLabels;
import static net.imglib2.algorithm.metrics.segmentation.SegmentationMetricsTest.getLabelingSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LazySEGMetricsTest
{

	private final static double delta = 0.00001;

	/**
	 * Test that passing a ground-truth image with intersecting labels throws an exception.
	 */
	@Test( expected = UnsupportedOperationException.class )
	public void testException()
	{
		final Img< IntType > img = ArrayImgs.ints( exampleIndexArray, exampleIndexArrayDims );
		final ImgLabeling< String, IntType > labeling = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleIntersectingLabels ) );
		final ImgLabeling< String, IntType > labeling2 = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleNonIntersectingLabels ) );

		new LazySEGMetrics().addTimePoint( labeling, labeling2 );
	}

	/**
	 * Test that passing a predicted image with intersecting labels throws an exception.
	 */
	@Test( expected = UnsupportedOperationException.class )
	public void testException2()
	{
		final Img< IntType > img = ArrayImgs.ints( exampleIndexArray, exampleIndexArrayDims );
		final ImgLabeling< String, IntType > labeling = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleNonIntersectingLabels ) );
		final ImgLabeling< String, IntType > labeling2 = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleIntersectingLabels ) );

		new LazySEGMetrics().addTimePoint( labeling, labeling2 );
	}

	/**
	 * Test that passing images without intersecting labels throws no exception.
	 */
	@Test
	public void testNoException()
	{
		final Img< IntType > img = ArrayImgs.ints( exampleIndexArray, exampleIndexArrayDims );
		final ImgLabeling< String, IntType > labeling = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleNonIntersectingLabels ) );
		final ImgLabeling< String, IntType > labeling2 = ImgLabeling.fromImageAndLabelSets( img, getLabelingSet( exampleNonIntersectingLabels ) );

		new LazySEGMetrics().addTimePoint( labeling, labeling2 );
	}

	/**
	 * Test score without adding any image.
	 */
	@Test
	public void testNoImage()
	{
		LazySEGMetrics metrics = new LazySEGMetrics( );
		assertEquals( Double.NaN, metrics.computeScore(), delta );
	}

	/**
	 * Test metrics values when either or both ground-truth and prediction image are empty.
	 */
	@Test
	public void testEmpty()
	{
		long[] dims = { 64, 64 };
		final Img< IntType > nonEmpty = ArrayImgs.ints( dims );
		final Img< IntType > empty = ArrayImgs.ints( dims );

		// paint
		SegmentationMetricsHelper.paintRectangle( nonEmpty, 12, 28, 42, 56, 9 );

		//////////////////////////////////
		// Empty gt, non-empty prediction
		LazySEGMetrics metrics = new LazySEGMetrics();
		metrics.addTimePoint( empty, nonEmpty );
		metrics.addTimePoint( empty, nonEmpty );

		double score = metrics.computeScore();
		assertEquals( Double.NaN, score, delta );

		//////////////////////////////////
		// Non-empty gt, empty prediction
		metrics = new LazySEGMetrics();
		metrics.addTimePoint( nonEmpty, empty );
		metrics.addTimePoint( nonEmpty, empty );

		score = metrics.computeScore();
		assertEquals( 0., score, delta );

		//////////////////////////////////
		// Empty gt, empty prediction
		metrics = new LazySEGMetrics();
		metrics.addTimePoint( empty, empty );
		metrics.addTimePoint( empty, empty );

		score = metrics.computeScore();
		assertEquals( Double.NaN, score, delta );
	}

	/**
	 * Test metrics values when two labels are overlapping, in particular depending on
	 * the threshold. This test uses XY images only (single frame, slice).
	 */
	@Test
	public void testOverlapping()
	{
		long[] dims = { 32, 32 };
		final Img< IntType > groundtruth = ArrayImgs.ints( dims );
		final Img< IntType > prediction = ArrayImgs.ints( dims );

		int[] gtRect1 = { 2, 2, 11, 11 };
		int[] predRect1 = { 6, 6, 15, 15 };

		int[] gtRect2 = { 15, 15, 20, 20 };
		int[] predRect2 = { 15, 16, 21, 21 };

		// Paint overlapping labels
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 5 );

		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 8 );

		// Calculate IoUs
		double iou1 = getSEGBetweenRectangles( gtRect1, predRect1 );
		double iou2 = getSEGBetweenRectangles( gtRect2, predRect2 );
		double[] ious = { iou1, iou2 };

		// Check the metrics versus the expected values
		LazySEGMetrics lazyMetrics = new LazySEGMetrics();
		lazyMetrics.addTimePoint( groundtruth, prediction );

		final double score = getScore( ious );
		final double lazyScore = lazyMetrics.computeScore();

		assertEquals( score, lazyScore, delta );
	}

	/**
	 * Test metrics values for a Z stack (XYZ), where labels are considered 3D, with respect to the threshold.
	 * In this case, the 3D labels are split between two slices with an empty slice in between.
	 */
	@Test
	public void testXYZ()
	{
		long[] dims = { 32, 32, 3 };
		final Img< IntType > groundtruth = ArrayImgs.ints( dims );
		final Img< IntType > prediction = ArrayImgs.ints( dims );

		int[] gtRect1 = { 2, 2, 11, 11 };
		int[] predRect1 = { 6, 6, 15, 15 };

		int[] gtRect2 = { 15, 15, 20, 20 };
		int[] predRect2 = { 15, 16, 21, 20 };

		// Paint rectangles on the first slice
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 0, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 0, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 0, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 0, 8 );

		// Paint rectangle with the same label on the last slice
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 2, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 2, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 2, 8 );

		// Rectangles with same label on all slices belong to the same 3D label
		double iou1 = getSEGBetweenRectangles( gtRect1, predRect1, 2 ); // calculates IoU for a rectangular volume
		double iou2 = getSEGBetweenRectangles( gtRect2, predRect2, 2 );
		double[] ious = { iou1, iou2 };

		// Check the metrics versus the expected values
		LazySEGMetrics lazyMetrics = new LazySEGMetrics();
		lazyMetrics.addTimePoint( groundtruth, prediction );

		final double score = getScore( ious );
		final double lazyScore = lazyMetrics.computeScore();

		assertEquals( score, lazyScore, delta );
	}

	/**
	 * Test metrics values for a movie (XYT, where T is nonetheless the 4th dimension), where labels
	 * are considered 2D, with respect to the threshold. In this case, the 2D labels are split between
	 * two time frames with an empty frame in between. The labels on each time frames have the same
	 * value in order to check that they are not aggregated.
	 */
	@Test
	public void testXYT()
	{
		long[] dims = { 32, 32, 1, 3 };
		final Img< IntType > groundtruth = ArrayImgs.ints( dims );
		final Img< IntType > prediction = ArrayImgs.ints( dims );

		int[] gtRect1 = { 2, 2, 11, 11 };
		int[] predRect1 = { 6, 6, 13, 14 };

		int[] gtRect2 = { 15, 15, 20, 20 };
		int[] predRect2 = { 15, 15, 22, 22 };

		int[] gtRect3 = { 4, 5, 11, 14 };
		int[] predRect3 = { 4, 5, 11, 15 };

		// paint rectangles in the first frame
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 0, 0, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 0, 0, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 0, 0, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 0, 0, 6 );

		// paint other rectangles in the last frame
		// We use same labels to test if they are not considered 3D
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 0, 2, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 0, 2, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect3, 0, 2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect3, 0, 2, 6 );

		// Calculate the IoUs
		double iou1 = getSEGBetweenRectangles( gtRect1, predRect1 );
		double iou2 = getSEGBetweenRectangles( gtRect2, predRect2 );
		double iou3 = getSEGBetweenRectangles( gtRect3, predRect3 );
		double[] ious = { iou1, iou2, iou2, iou3 };

		// Check the metrics versus the expected values
		LazySEGMetrics lazyMetrics = new LazySEGMetrics();
		for ( int i = 0; i < 3; i++ )
		{
			final RandomAccessibleInterval< IntType > gt = Views.hyperSlice( groundtruth, 3, i );
			final RandomAccessibleInterval< IntType > p = Views.hyperSlice( prediction, 3, i );

			lazyMetrics.addTimePoint( gt, p );
		}

		final double score = getScore( ious );
		final double lazyScore = lazyMetrics.computeScore();

		assertEquals( score, lazyScore, delta );
	}

	/**
	 * Test metrics values for a 4D movie (XYZT), where labels are considered 2D, with respect to the
	 * threshold. Here, we paint 3D boxes in the firs time frame with an empty slice in the middle of
	 * the boxes. On the last frame, we do the same with different 3D boxes. The boxes in the first
	 * time frame have the same label as in the last time frame, to make sure that they are not
	 * aggregated. We put an empty time frame in between.
	 */
	@Test
	public void testXYZT()
	{
		long[] dims = { 32, 32, 3, 3 };
		final Img< IntType > groundtruth = ArrayImgs.ints( dims );
		final Img< IntType > prediction = ArrayImgs.ints( dims );

		int[] gtRect1 = { 2, 2, 11, 11 };
		int[] predRect1 = { 6, 6, 13, 14 };

		int[] gtRect2 = { 15, 15, 20, 20 };
		int[] predRect2 = { 15, 15, 22, 22 };

		int[] gtRect3 = { 4, 5, 11, 14 };
		int[] predRect3 = { 4, 5, 11, 15 };

		// paint 3D boxes on first frame
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 0, 0, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 0, 0, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 0, 0, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 0, 0, 6 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 2, 0, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 2, 0, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 2, 0, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 2, 0, 6 );

		// paint last frame (we use same labels to test if they get confused)
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 0, 2, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 0, 2, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect3, 0, 2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect3, 0, 2, 6 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 2, 2, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 2, 2, 5 );
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect3, 2, 2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect3, 2, 2, 6 );

		// Calculate IoUs
		double iou1 = getSEGBetweenRectangles( gtRect1, predRect1, 2 );
		double iou2 = getSEGBetweenRectangles( gtRect2, predRect2, 2 );
		double iou3 = getSEGBetweenRectangles( gtRect3, predRect3, 2 );
		double[] ious = { iou1, iou2, iou2, iou3 };

		// Check the metrics versus the expected values
		LazySEGMetrics lazyMetrics = new LazySEGMetrics();
		for ( int i = 0; i < 3; i++ )
		{
			final RandomAccessibleInterval< IntType > gt = Views.hyperSlice( groundtruth, 3, i );
			final RandomAccessibleInterval< IntType > p = Views.hyperSlice( prediction, 3, i );

			lazyMetrics.addTimePoint( gt, p );
		}

		final double score = getScore( ious );
		final double lazyScore = lazyMetrics.computeScore();

		assertEquals( score, lazyScore, delta );
	}

	/**
	 * Showcase a multithreading use of the LazySEGMetrics
	 *
	 * @throws InterruptedException
	 */
	public void testMultithreding() throws InterruptedException
	{
		int M = 10;
		int N = 4 * 100;

		long[] dims = { 512, 512 };
		final Img< IntType > groundtruth = ArrayImgs.ints( dims );
		final Img< IntType > prediction = ArrayImgs.ints( dims );

		int[] gtRect1 = { 2, 2, 11, 11 };
		int[] predRect1 = { 6, 6, 15, 15 };

		int[] gtRect2 = { 15, 15, 20, 20 };
		int[] predRect2 = { 15, 16, 21, 21 };

		// Paint overlapping labels
		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect1, 9 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect1, 5 );

		SegmentationMetricsHelper.paintRectangle( groundtruth, gtRect2, 2 );
		SegmentationMetricsHelper.paintRectangle( prediction, predRect2, 8 );

		double start, end, lazyResults = 0.;
		double timeLazy = 0;
		for ( int k = 0; k < M; k++ )
		{
			start = System.currentTimeMillis();

			LazySEGMetrics lazyMetrics = new LazySEGMetrics();
			for ( int i = 0; i < N; i++ )
			{
				lazyMetrics.addTimePoint( groundtruth, prediction );
			}
			lazyResults = lazyMetrics.computeScore();

			end = System.currentTimeMillis();

			if ( k > 0 )
				timeLazy += ( end - start ) / 100.;

		}

		double timeMultiLazy = 0., multiLazyResults = 0.;
		for ( int k = 0; k < M; k++ )
		{
			start = System.currentTimeMillis();

			final LazySEGMetrics lazyMetrics = new LazySEGMetrics();
			Loader l1 = new Loader( lazyMetrics, groundtruth, prediction, N / 4 );
			Loader l2 = new Loader( lazyMetrics, groundtruth, prediction, N / 4 );
			Loader l3 = new Loader( lazyMetrics, groundtruth, prediction, N / 4 );
			Loader l4 = new Loader( lazyMetrics, groundtruth, prediction, N / 4 );

			Thread t1 = new Thread( l1 );
			Thread t2 = new Thread( l2 );
			Thread t3 = new Thread( l3 );
			Thread t4 = new Thread( l4 );

			t1.start();
			t2.start();
			t3.start();
			t4.start();

			while ( !l1.isDone() || !l2.isDone() || !l3.isDone() || !l4.isDone() )
			{
				Thread.sleep( 1 );
			}

			multiLazyResults = lazyMetrics.computeScore();

			end = System.currentTimeMillis();

			if ( k > 0 )
				timeMultiLazy += ( end - start ) / 100.;
		}

		System.out.println( timeLazy + " vs " + timeMultiLazy );

		assertTrue( 2 * timeMultiLazy < timeLazy );

		assertEquals( lazyResults, multiLazyResults, delta );
	}

	private class Loader implements Runnable
	{

		private AtomicInteger counter;

		private final int max;

		private final LazySEGMetrics metrics;

		private final RandomAccessibleInterval< IntType > gtRai, pRai;

		public Loader( final LazySEGMetrics metrics, RandomAccessibleInterval< IntType > gt, RandomAccessibleInterval< IntType > pred, int max )
		{
			this.counter = new AtomicInteger( 0 );
			this.metrics = metrics;
			this.max = max;

			this.gtRai = ArrayImgs.ints( gt.dimensionsAsLongArray() );
			LoopBuilder.setImages( gtRai, gt ).forEachPixel( ( o, i ) -> o.set( i ) );

			this.pRai = ArrayImgs.ints( gt.dimensionsAsLongArray() );
			LoopBuilder.setImages( pRai, pred ).forEachPixel( ( o, i ) -> o.set( i ) );
		}

		@Override
		public void run()
		{
			while ( counter.get() < max )
			{
				metrics.addTimePoint( gtRai, pRai );
				counter.incrementAndGet();
			}
		}

		public boolean isDone()
		{
			return counter.get() == max;
		}
	}

	private static double getScore( final double[] ious )
	{
		// Compute expected values
		double nGT = ious.length;
		double sumIoU = Arrays.stream( ious ).sum();

		return sumIoU / nGT;
	}
}