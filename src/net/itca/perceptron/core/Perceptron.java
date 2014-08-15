package net.itca.perceptron.core;

public class Perceptron 
{
	private int[][] trainingData = new int[5][];
	private int generations = 1000;
    private double learningRate = 0.075d; // learningRate
    private double targetError = 0.0d;
	private double bestBias = 0.0d;
    public Perceptron()
	{
		// (Last bit represents whether the data is a B (1) or is not a B (0)
	    trainingData[0] = new int[] { 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,   0 };  // 'A'
	    trainingData[1] = new int[] { 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,   1 };  // 'B'
	    trainingData[2] = new int[] { 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,   0 };  // 'C'
	    trainingData[3] = new int[] { 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,   0 };  // 'D'
	    trainingData[4] = new int[] { 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1,   0 };  // 'E'
	    drawLetters(); // Draw all letters in console.
	    System.out.println("");
	    System.out.println("finding the best weights and bias for 'B'\nStarting training...");
	    double targetError = 0.0d;
	    double[] bestWeights = this.FindBestWeights(trainingData, generations, learningRate, targetError);
	    System.out.println("Training complete! :)");
	    System.out.println("Best weights and bias are: \n");
	    this.DrawBestWeightsAndBias(bestWeights);
	    System.out.println("\nBest bias: " + bestBias);
	    double totalError = TotalError(bestWeights, bestBias);
	    System.out.println("Total error after training: " + totalError);
	    
	    // play around with this, damage the B in some positions and see if it's recognized! :)
	    int[] toPredict = new int[] { 1, 1, 0, 0,   1, 0, 0, 1,   1, 1, 1, 0,   1, 0, 0, 1,   0, 1, 1, 0 };  // damaged 'B' 
	    System.out.println("\nPredicting if it's a 'B' (1 = yes, 0 = no) for the following data");
	    draw(toPredict);
	    
	    int prediction = Predict(toPredict, bestWeights, bestBias);
	    System.out.println("\nOutcome is: " + prediction);
	    System.out.println("\n" + (prediction == 1 ? "is recognized as B" : "is not recognized as B"));
	    
	}
    
    private double[] FindBestWeights(int[][] data, int maxGen, double lr, double targetError)
    {
    	int weightLength = data[0].length-1;
    	double[] weights = new double[weightLength];
    	double bias = 0.05d;
    	double totalError = Double.MAX_VALUE;
    	int currentGeneration = 0;
    	
    	while(currentGeneration < maxGen && totalError > targetError)
    	{
    		for(int entry = 0; entry < trainingData.length;entry++)
    		{
    			int desired = trainingData[entry][trainingData[entry].length-1];
    			int out = ComputeOutput(trainingData[entry],weights,bias);
    			int delta = desired - out;
    			
    			for(int currentWeight = 0; currentWeight < weights.length; currentWeight++)
    			{
    				weights[currentWeight] = weights[currentWeight] + (learningRate * delta * data[entry][currentWeight]);
    			}
    			bias = bias + (learningRate * delta);
    		}
    		totalError = TotalError(weights, bias);
    		currentGeneration++;
    	}
    	bestBias = bias;
    	return weights;
    }
	
    private double TotalError(double[] weights, double bias)
    {
    	double sigma = 0.0;
    	
    	for(int entry = 0; entry < trainingData.length;entry++)
    	{
    		int target = trainingData[entry][trainingData[entry].length-1]; // Target is our desired value
    		int out = ComputeOutput(trainingData[entry],weights,bias);
    		sigma += (target - out) * (target - out);
    	}
    	
    	return sigma * 0.5;
    }
    
    private int ComputeOutput(int[] trainVector, double[] weights, double bias)
    {
    	double dotProduct = 0.0;
    	for(int entry = 0; entry < trainVector.length-1 /* last bit indicates if it's a B or not */; entry++)
    	{
    		dotProduct += (trainVector[entry]*weights[entry]);
    	}
    	dotProduct += bias;
    	return StepFunction(dotProduct);
    }
    
    private int StepFunction(double x)
    {
    	return (x > 0.5 ? 1 : 0);
    }
	
	private void drawLetters()
	{
		for(int i = 0; i < 5; i++)
		{
			draw(trainingData[i]);
		}
	}
	// Draws the letters the vectors represent.
	public void draw(int[] vector)
	{
		System.out.println(""); // clear line above print
		for(int bit = 0; bit < 20; bit++)
		{
			if(bit % 4 == 0)
			{
				System.out.println("");
			}
			if(vector[bit]== 0)
			{
				System.out.print(" ");
			}
			else
			{
				System.out.print("1");
			}
		}
		System.out.println(""); // clear line beneath print
	}
	
	private void DrawBestWeightsAndBias(double[] vector)
	{
		for(int bwab = 0; bwab < vector.length; bwab++)
		{
			if(bwab > 0 && bwab % 4 == 0)
			{
				System.out.println("");
			}
			System.out.print(vector[bwab] + "+0.000;-0.000"+ " ");
		}
		System.out.println();
	}
	
	/* Prediction logic */
	
	private int Predict(int[] data, double[] bestWeights, double bestBias)
	{
		double dotProduct = 0.0d;
		for(int entry = 0; entry < data.length; entry++)
		{
			dotProduct += (data[entry] * bestWeights[entry]);
		}
		dotProduct += bestBias;
		return StepFunction(dotProduct);
	}
}
