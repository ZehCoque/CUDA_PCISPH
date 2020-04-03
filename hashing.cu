////defining 3 prime numbers
//
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <iostream>
//#include "device_functions.cu"
//#include "common.cuh"
//
//bool isPrime(int n)
//{
//    // Corner cases
//    if (n <= 1)  return false;
//    if (n <= 3)  return true;
//
//    // This is checked so that we can skip
//    // middle five numbers in below loop
//    if (n%2 == 0 || n%3 == 0) return false;
//
//    for (int i=5; i*i<=n; i=i+6)
//        if (n%i == 0 || n%(i+2) == 0)
//           return false;
//
//    return true;
//}
//
//// Function to return the smallest
//// prime number greater than N
//int nextPrime(int N)
//{
//
//    // Base case
//    if (N <= 1)
//        return 2;
//
//    int prime = N;
//    bool found = false;
//
//    // Loop continuously until isPrime returns
//    // true for a number greater than n
//    while (!found) {
//        prime++;
//
//        if (isPrime(prime))
//            found = true;
//    }
//
//    return prime;
//}
//
//class Hash
//{
//    int hashtable_size;    // No. of buckets
//    int p1 = 73856093;
//    int p2 = 19349669;
//    int p3 = 83492791;
//
//
//public:
//    int **hash_table;
//    Hash(int V);  // Constructor
//
//    // inserts a key into hash table
//    __device__ void insertItem(vec3d point,int point_id,float h);
//
//    // deletes a key from hash table
//    __device__ void deleteItem(int key);
//
//    // hash function to map values to key
//    __device__ int hashFunction(vec3d point,float h);
//
//    __device__ void displayHash();
//};
//
//Hash::Hash(int b)
//{
//    this->hashtable_size = b;
//
//    this->hash_table = (int**)malloc(b*sizeof(int));
//
//    for (int i = 0;i < b;i++){
//        hash_table[i] = (int*)malloc(30*sizeof(int));
//    }
//
//    gpuErrchk(cudaMallocManaged(&hash_table, b * 30 * sizeof(int)));
//
//}
//
//__device__ int Hash::hashFunction(vec3d point,float h) {
//
//    int r_x,r_y,r_z;
//
//    r_x = static_cast<int>(floor(point.x/h));
//    r_y = static_cast<int>(floor(point.y/h));
//    r_z = static_cast<int>(floor(point.z/h));
//    //printf("%d\n",(r_x ^ r_y ^ r_z) % hashtable_size);
//    return ((r_x ^ r_y ^ r_z) % hashtable_size);
//}
//
//__device__ void Hash::insertItem(vec3d point,int point_id,float h)
//{
//
//    int index = hashFunction(point,h);
//    //hash_table[index][0] = point_id;
//
//}
//
//// __device__ void Hash::deleteItem(int key)
//// {
////   // get the hash index of key
////   int index = hashFunction(key);
//
////   // find the key in (inex)th list
////   std::list <int> :: iterator i;
////   for (i = table[index].begin();
////            i != table[index].end(); i++) {
////     if (*i == key)
////       break;
////   }
//
////   // if key is found in hash table, remove it
////   if (i != table[index].end())
////     table[index].erase(i);
//// }
//
//// function to display hash table
//// __device__ void Hash::displayHash() {
////   for (int i = 0; i < BUCKET; i++) {
////     std::cout << i;
////     for (auto x : table[i])
////       std::cout << " --> " << x;
////     std::cout << std::endl;
////   }
//// }
//
//
//__global__ void hashParticlePositions(vec3d* points, float h,Hash hash){
//
//    int index = getGlobalIdx_1D_1D();
//
//    hash.insertItem(points[index],index,h);
//
//}
//
//int main(){
//
//    float h = 2.45;
//
//    vec3d* points = new vec3d[2];
//    points[0].x = 0.252;
//    points[0].y = 1.524;
//    points[0].z = 5.45;
//
//    points[1].x = 6.545;
//    points[1].y = 0;
//    points[1].z = 1.7;
//
//    int hashtable_size = nextPrime(200);
//
//    Hash hash(hashtable_size);
//
//    // for (int i = 0; i < 2; i++)  {
//    //     hash.insertItem(points[i],i,h);
//    // }
//
//
//    hashParticlePositions<<<64,64>>>(points,h,hash);
//    // for (int i = 0;i<2;i++){
//    //printf("%d\n",hash.hash_table[0][0]);
//    // }
//
//
//}