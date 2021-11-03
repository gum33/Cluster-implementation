using System;
using System.Linq;
using System.IO;
using System.Text;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Assignment3
{
    //Stores the mean distance of data and the indeces of the data point in this cluster
    class Cluster 
    {
        public List<List<int>> indeces { get; } //indeces in the corresponding dataset that are in this cluster
        public List<List<double[]>> locations { get; } //the locations of the datapoints in clusters
        public List<double[]> AvgLocation { get;} //Average location of clusters
        public int Count { get; set; } //track current number of clusters

        //Initialize cluster object with a dataset of location and matching indeces
        public Cluster(List<List<int>> indeces, List<List<double[]>> locations)
        {
            this.indeces = indeces; 
            this.locations = locations;
            this.Count = locations.Count;
            AvgLocation = new List<double[]>();
            foreach(List<double[]> loc in locations) //initial average location is the location passed to the cluster
            {
                AvgLocation.Add(loc[0]);
            }
        }

        //Combines two points into a cluster, i
        public void MergeClusters(int i, int j)
        {
            if (j >= Count || i >= Count)
                throw new ArgumentException("Indeces must be lower than number of clusters");
            locations[i].AddRange(locations[j]);
            locations.RemoveAt(j);
            indeces[i].AddRange(indeces[j]);
            indeces.RemoveAt(j);
            AvgLocation.RemoveAt(j);
            UpdateAvgLocation(i);
            --Count;
        }

        //Add data point j with values in loc to cluster number i
        public void AddtoCluster(int i, int j, double[] loc)
        {
            if (i > Count)
                throw new ArgumentException("i must be lower than number of clusters");
            locations[i].Add(loc);
            indeces[i].Add(j);
            UpdateAvgLocation(i);
        }

        //Remove datapoint with index i and at location loc from the clusterset
        public void Remove(int i, double[] loc)
        {
          
            int j;
            bool incluster = false; //to check if i is currently in any cluster
            for (j = 0; j < Count; j++)
            {
                if (indeces[j].Contains(i))
                {
                    incluster = true;
                    break;
                }
            }
            if (incluster) //If its in some cluster remove it otherwise do nothing
            {
                locations[j].Remove(loc);
                indeces[j].Remove(i);
                UpdateAvgLocation(j);
            }
        }

        //Updates the avglocationn of a cluster after merging or removing datapoints
        private void UpdateAvgLocation(int i)
        {
            //Current dimension of the object
            int dimension = locations[i][0].Length;
            double[] avg = new double[dimension];
            for (int j = 0; j < dimension; j++)
            { 
                for (int k = 0; k < locations[i].Count; k++)
                {
                    avg[j] += locations[i][k][j] / locations[i].Count; //the average location
                }
            }
            AvgLocation[i] = avg; //replace old avg location
   
        }

        //Way of presenting clusters on the standard output for exploring and deubbing
        public override string ToString()
        {
            string s = "";
            for(int i = 0; i<Count; i++)
            {
                //prints in each line the indeces and average location of a cluster
                s += string.Format("({0}): ({1})\n", string.Join("," , indeces[i]), string.Join(", ", AvgLocation[i]));
            }
            return s;
        }

        //Get cluster wtith index i in string form
        //Returns indeces and average location with delimeter, last d elements are the location
        public string At(int i)
        {
            return string.Format("{0} , {1}\n", string.Join(",", indeces[i]), string.Join(", ", AvgLocation[i]));
        }
    }

    //A method of creating cluster with initilzalie nr of clusters and make the points join the closest cluster until it converges
    class ConvergingCluster
    {
        private Func<double[], List<double[]>, double[]> distance; //Function used to calculate distances
        public List<double[]> data; //Original dataset
        public Cluster Clusters { get; set; } //Collection of the clusters

        //initialize the class
        public ConvergingCluster(List<double[]> data, Func<double[], List<double[]>, double[]> distance)
        {
            this.distance = distance;
            this.data = data;
        }

        //Createclusters based on the original data based on the nrofClusters
        public Cluster CreateCluster(int nrofClusters)
        {
            if (nrofClusters < 1 || nrofClusters > data.Count)
                throw new ArgumentException("Number of clusters must be >0 and less than number of datapoints");

            
            List<int> randomValues = Shuffle(); //shuffle the indeces then we take first n to be cluster centers.
            List<List<int>> indeces = new List<List<int>>();
            List<List<double[]>> locs = new List<List<double[]>>();
            //initialize the clusters with just the random cluster centers
            for(int i = 0; i<nrofClusters; i++)
            {
                indeces.Add(new List<int> { randomValues[i] });
                locs.Add(new List<double[]> { data[randomValues[i]] });
            }

            Clusters = new Cluster(indeces, locs);

            
            double minDist;
            int minIndex = 0, change = 1;
            //While there are still changes happening in the cluster continue creating
            while (change != 0)
            {
                change = 0;
                for (int i = 0; i < data.Count; i++)
                {
                    int val = randomValues[i]; //val is the index of the datapoint we look at first
                    minDist = double.MaxValue;
                    //gets the distance of clusters centers from datapoint val
                    double[] distfromI = distance(data[val], Clusters.AvgLocation); 
                    for (int j = 0; j < distfromI.Length; j++)
                    {
                        if (distfromI[j] < minDist) //finds the closest cluster and saves index of cluster and the distance
                        {
                            minDist = distfromI[j];
                            minIndex = j;
                        }
                    }
                    //If the closest cluster is not the one it is currently in, remove from old cluster add to new
                    //else it is in the best possible cluster and we do nothing
                    if (!Clusters.indeces[minIndex].Contains(val))
                    {
                        Clusters.Remove(val, data[val]);
                        Clusters.AddtoCluster(minIndex, val, data[val]);
                        ++change; //Add change meaning it has not converged
                    }

                }
            }
            return Clusters;

            
        }

        //Fisher-Yates shuffle
        private List<int> Shuffle()
        {
            var result = Enumerable.Range(0, data.Count).ToList();
            var r = new Random();
            //Goes backward in a list and randomly swaps with another element infront in the list
            for (int i = data.Count; i > 1; i--)
            {
                int j = r.Next(i); //random index up to i
                int t = result[j]; 
                result[j] = result[i - 1];
                result[i - 1] = t;
            }
            return result;
        }
    }

    class GreedyCluster
    {
        private Func<List<double[]>,double[,] > distance; //distance function
        private List<double[]> data; //original data
        public Cluster Clusters { get; set; }//Store the clusters

        public GreedyCluster(List<double[]> data, Func<List<double[]>, double[,]> distance) 
        {
            this.distance = distance;
            this.data = data;
            List<List<int>> indeces = new List<List<int>>();
            List<List<double[]>> locs = new List<List<double[]>>();
            for (int i = 0; i < data.Count; i++)
            {
                indeces.Add(new List<int> {i });
                locs.Add(new List<double[]> {data[i]});
            }
            Clusters = new Cluster(indeces, locs);
        }

        public Cluster CreateCluster(int nrofClusters)
        {
            if (nrofClusters < 1 || nrofClusters > Clusters.Count)
                throw new ArgumentException("Number of clusters must be >0 and less than number of datapoints");

            double[,] DistanceMatrix = distance(Clusters.AvgLocation);


            double minDistance,  mindDistance2nd = int.MaxValue; //Store closest points and 
            int j = 0; int k=0;
            Tuple<double, int> minvals; double[] row;
            while (DistanceMatrix.GetLength(0) > nrofClusters)
            {
                minDistance = int.MaxValue;
                for (int i = 0; i < Clusters.Count-1; i++) 
                {
                    row = GetRow(DistanceMatrix,i); //Get distance from point i to points that we havent compared with
                    minvals = ClosestPoint(row,i); // get distance to closest point and index of that point
                    if (minvals.Item1 < minDistance)
                    {
                        minDistance = minvals.Item1;
                        j = minvals.Item2;
                        k = i;
                    }
                }

                Clusters.MergeClusters(k, j);
                DistanceMatrix = distance(Clusters.AvgLocation);
            }

            return Clusters;
        }

        //Get the shortest distance to another poiint
        private Tuple<double, int> ClosestPoint(double[] distRow, int rownr)
        {
            double minDistance = double.MaxValue;
            int minindex = 0;
            for (int k = rownr+1; k < distRow.Length; ++k)
            {
                if (distRow[k] < minDistance)
                {
                    minDistance = distRow[k];
                    minindex = k;
                }
            }
            return Tuple.Create(minDistance, minindex);
        }

        //Get one row of a double[,] array
        private double[] GetRow(double[,] matrix, int rowNumber)
        {
            double[] row = new double[matrix.GetLength(1)];
            for(int i = 0; i<row.Length; i++)
            {
                row[i] = matrix[rowNumber, i];
            }
            return row;
        }

    }

    //Class for calculating distances
    static class Distance
    {
        //Method to print matrices in a clear way for debugging
        public static void printmatrix(double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)

                {
                    Console.Write("{0}, ", matrix[i, j]);
                }
                Console.WriteLine("");
            }
        }

        //Euclidean distance of a vector from points in a data matrix
        public static double[] Euclidean(double[] vector, List<double[]> data)
        {
            int d = data[0].Length; //Dimensions
            int n = data.Count; //nr of datapoints
            double[] dist = new double[n]; //distance vector
            for (int j = 0; j < n; j++)
            {
                for (int l = 0; l < d; l++)
                {
                    dist[j] += Math.Pow(vector[l] - data[j][l], 2);
                }
                dist[j] = Math.Sqrt(dist[j]);
            }
            return dist;
        }

        //Given a data set returns a distance matrix
        public static double[,] Euclidean(List<double[]> data)
        {
            int d = data[0].Length; //Dimensions
            int n = data.Count; //nr of datapoints
            double[,] dist = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    for (int l = 0; l < d; l++)
                    {
                        dist[i, j] += Math.Pow(data[i][l] - data[j][l], 2);

                    }
                    dist[i, j] = Math.Sqrt(dist[i, j]);
                    dist[j, i] = dist[i, j]; //use that distance matrix is symmetric
                }

            }
            return dist;
        }


        //Returns a distance vector giving the manhattan distance of vector from the points in data
        public static double[] Manhattan(double[] vector, List<double[]> data)
        {
            int n = data.Count; //nr of datapoints
            int d = data[0].Length; //dimension of data
            double[] dist = new double[n];
    
            for (int j = 0; j < n; j++)
            {
                for (int l = 0; l < d; l++)
                {
                    dist[j] += Math.Abs(vector[l] - data[j][l]);
                }

            }
         
            return dist;
        }

        //returns a full distance matrix using Manhattan measure
        public static double[,] Manhattan(List<double[]> data)
        {
            int n = data.Count; //nr of datapoints
            int d = data[0].Length; //dimension of data
            double[,] dist = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    for (int l = 0; l < d; l++)
                    {
                        dist[i, j] += Math.Abs(data[i][l] - data[j][l]);
                    }
                    dist[j, i] = dist[i, j];

                }
            }
            return dist;
        }


        //Returns a distance vector from point to all data
        public static double[] Mahalanobis(double[] point, List<double[]> data)
        {
            var M = Matrix<double>.Build; //matrix builder
            List<double[]>_mydata = new List<double[]>(data); //make a copy of the data
            _mydata.Add(point); //add the point to the end of the data copy
            double[,] datarray = To2dArray(_mydata); //turn data into double[,]
            Matrix<double> matrixdata = M.DenseOfArray(datarray); //put the data into matrix form
            Matrix<double> S = M.DenseOfArray(SampleCovar(_mydata)); //Calculate inverse of covariance matrix
            Matrix<double> Sinv = S.Inverse(); // inverse of covar matrix
            int n = _mydata.Count; //nr of samples
            Vector<double> dif;
            double[] dist = new double[n-1];
 
            for (int j = 0; j < n-1; j++)
            {
                //use definition of Mahalanobis to get a vector of distances
                dif = matrixdata.Row(n-1) - matrixdata.Row(j);
                dist[j] = Math.Sqrt(dif * Sinv.Multiply(dif));
                dist[j] = dist[j];
            }


            return dist;
        }

        //Distnace matrix using Mahalanobis works same as method above but calculates every distance
        public static double[,] Mahalanobis(List<double[]> data)
        {
            var M = Matrix<double>.Build;
            double[,] datarray = To2dArray(data);
            Matrix<double> matrixdata = M.DenseOfArray(datarray);
            Matrix<double> S =  M.DenseOfArray(SampleCovar(data));
            Matrix<double> Sinv = S.Inverse();
            int n = data.Count; //nr of samples
            Vector<double> dif;
            double[,] dist = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = i+1 ; j < n; j++)
                {
                   
                    dif = matrixdata.Row(i) - matrixdata.Row(j);
                    dist[i, j] = Math.Sqrt(dif*Sinv.Multiply(dif)) ;
                    dist[j, i] = dist[i, j]; //use of simmetry
                }

            }
            return dist;
        }

        //change  List of doubles to a 2d array
        private static double[,] To2dArray(List<double[]> list)
        {
            int n = list.Count; //nr of datapoints
            int d = list[0].Length; // dimensions
            double[,] array = new double[n, d];
            //fill the 2d array
            for (int r = 0; r < n; r++)
                for (int c = 0; c < d; c++)
                    array[r, c] = list[r][c];
            return array;
        }

        //calculate covariance matrix of agiven dataset
        private static double[,] SampleCovar(List<double[]> data)
        {
            int d = data[0].Length; //dimension
            int n = data.Count; //nr of data points
            double[,] Scvar = new double[d, d];
            double[] Smean = SampleMean(data);
            for (int i = 0; i < d; i++)
            {
                for (int j = i ; j < d; j++)
                {
                    for (int l = 0; l < n; l++)
                    {
                        Scvar[i, j] += (data[l][i]-Smean[i]) * (data[l][j]-Smean[j]);
                    }
                    Scvar[i, j] = (Scvar[i, j]) / (d-1);
                    Scvar[j, i] = Scvar[i, j];
                }
            }
            return Scvar;
        }

        //Calculate mean position of a sample.
        private static double[] SampleMean(List<double[]> data)
        {
            int n = data.Count; //number of samples
            int d = data[0].Length; //dimension
            double[] mean = new double[d];
            for (int i = 0; i < d; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    mean[i] += data[j][i];
                }
                mean[i] = mean[i] / (double)d;
            }
            return mean;
        }
    }

    class Program
    {
        //Function that does task33 and writes the clusters to csv files.
        public static void Task32()
        {
            string path = Directory.GetCurrentDirectory(); //gets the path of the compiled file
            string lowerpath = Path.GetFullPath(Path.Combine(path, "..", "..", "..", "..")); //goes up 4 levels in folders

            List<double[]> cluster2 = LoadDoubleCsv(lowerpath + "/cluster2.csv"); //read cluster2.csv must be saved in the same folder the .sln file
            List<double[]> cluster4 = LoadDoubleCsv(lowerpath + "/cluster4.csv");//read cluster4.csv must be saved in the same folder the .sln file


            ConvergingCluster CclusterEuclid2 = new ConvergingCluster(cluster2, Distance.Euclidean);
            ConvergingCluster CclusterManhattan2 = new ConvergingCluster(cluster2, Distance.Manhattan);
            ConvergingCluster CclusterMahalanobis2 = new ConvergingCluster(cluster2, Distance.Mahalanobis);

            WriteClusterCSV(CclusterEuclid2.CreateCluster(6), lowerpath + "/Ccluster2-Euclid.csv");
            WriteClusterCSV(CclusterManhattan2.CreateCluster(6), lowerpath + "/Ccluster2-Manhatt.csv");
            WriteClusterCSV(CclusterMahalanobis2.CreateCluster(6), lowerpath + "/Ccluster2-Mahal.csv");

            ConvergingCluster CclusterEuclid4 = new ConvergingCluster(cluster4, Distance.Euclidean);
            ConvergingCluster CclusterManhattan4 = new ConvergingCluster(cluster4, Distance.Manhattan);
            ConvergingCluster CclusterMahalanobis4 = new ConvergingCluster(cluster4, Distance.Mahalanobis);

            WriteClusterCSV(CclusterEuclid4.CreateCluster(6), lowerpath + "/Ccluster4-Euclid.csv");
            WriteClusterCSV(CclusterManhattan4.CreateCluster(6), lowerpath + "/Ccluster4-Euclid.csv");
            WriteClusterCSV(CclusterMahalanobis4.CreateCluster(6), lowerpath + "/Ccluster4-Mahal.csv");

            GreedyCluster GclusterEuclid2 = new GreedyCluster(cluster2, Distance.Euclidean);
            GreedyCluster GclusterManhattan2 = new GreedyCluster(cluster2, Distance.Manhattan);
            GreedyCluster GclusterMahalanobis2 = new GreedyCluster(cluster2, Distance.Mahalanobis);

            WriteClusterCSV(GclusterEuclid2.CreateCluster(6), lowerpath + "/Gcluster2-Euclid.csv");
            WriteClusterCSV(GclusterManhattan2.CreateCluster(6), lowerpath + "/Gcluster2-Manhatt.csv");
            WriteClusterCSV(GclusterMahalanobis2.CreateCluster(6), lowerpath + "/Gcluster2-Mahal.csv");

            GreedyCluster GclusterEuclid4 = new GreedyCluster(cluster4, Distance.Euclidean);
            GreedyCluster GclusterManhattan4 = new GreedyCluster(cluster4, Distance.Manhattan);
            GreedyCluster GclusterMahalanobis4 = new GreedyCluster(cluster4, Distance.Mahalanobis);

            WriteClusterCSV(GclusterEuclid4.CreateCluster(6), lowerpath + "/Gcluster4-Euclid.csv");
            WriteClusterCSV(GclusterManhattan4.CreateCluster(6), lowerpath + "/Gcluster4-Manhatt.csv");
            WriteClusterCSV(GclusterMahalanobis4.CreateCluster(6), lowerpath + "/Gcluster4-Mahal.csv");


        }
        public static void WriteClusterCSV(Cluster Clusters, string filePath)
        {

            string delimiter = ",";
            

            int length = Clusters.Count;
            StringBuilder sb = new StringBuilder();
            for (int index = 0; index < length; index++)
                sb.AppendLine(string.Join(delimiter, Clusters.At(index)));

            File.WriteAllText(filePath, sb.ToString());
        }

        //read csv file into List<double[]>
        public static List<double[]> LoadDoubleCsv(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath); //read all lines
            List<double[]> jaggedArray = new List<double[]>();
            for (int i = 1; i < lines.Length; i++)
            {
                //read each line and skip the index in first position and then convert to double
                string[] strArray = lines[i].Split(',').Skip(1).ToArray(); 
                double[] intArray = Array.ConvertAll(strArray, double.Parse);
                jaggedArray.Add(intArray);
            }

            return jaggedArray;
        }

        static void Main(string[] args)
        {
            List<double[]> data = new List<double[]>();
            data.Add(new double[] { 1.5, 1.21, 1.22 });
            data.Add(new double[] { 4, 3.5, 1.23 });
            data.Add(new double[] { 1.5, 1.2, 1.23 });
            data.Add(new double[] { 2.1, 2.2, 3 });
            data.Add(new double[] { 2.11, 2.2, 3.1});



            Task32();
            Console.WriteLine("Task32 complete");
            Console.ReadKey();
        }
    }
}
//File change
