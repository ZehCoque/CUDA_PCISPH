//defining 3 prime numbers
//#include<bits/stdc++.h> 

bool isPrime(int n)  
{  
    // Corner cases  
    if (n <= 1)  return false;  
    if (n <= 3)  return true;  
    
    // This is checked so that we can skip   
    // middle five numbers in below loop  
    if (n%2 == 0 || n%3 == 0) return false;  
    
    for (int i=5; i*i<=n; i=i+6)  
        if (n%i == 0 || n%(i+2) == 0)  
           return false;
    
    return true;
}  
  
// Function to return the smallest 
// prime number greater than N 
int nextPrime(int N) 
{ 
  
    // Base case 
    if (N <= 1) 
        return 2; 
  
    int prime = N; 
    bool found = false; 
  
    // Loop continuously until isPrime returns 
    // true for a number greater than n 
    while (!found) { 
        prime++; 
  
        if (isPrime(prime)) 
            found = true; 
    } 
  
    return prime; 
} 

class Hash 
{ 
    int BUCKET;    // No. of buckets 
    int p1 = 73856093;
    int p2 = 19349669;
    int p3 = 83492791;
    // Pointer to an array containing buckets 
    std::list<int> *table; 
public: 
    Hash(int V);  // Constructor 
  
    // inserts a key into hash table 
    __device__ void insertItem(vec3d point); 
  
    // deletes a key from hash table 
    __device__ void deleteItem(int key); 
  
    // hash function to map values to key 
    __device__ int hashFunction(vec3d point,float h) { 

        int r_x,r_y,r_z;

        r_x = static_cast<int>(floor(point.x/h));
        r_y = static_cast<int>(floor(point.y/h));
        r_z = static_cast<int>(floor(point.z/h));

        return ((r_x ^ r_y ^ r_z) % BUCKET); 
    } 
  
    __device__ void displayHash(); 
}; 
  
Hash::Hash(int b) 
{ 
    this->BUCKET = b; 
    table = new std::list<int>[BUCKET]; 
} 
  
__device__ void Hash::insertItem(vec3d point) 
{ 
    int index = hashFunction(point); 
    table[index].push_back(key);  
} 
  
__device__ void Hash::deleteItem(int key) 
{ 
  // get the hash index of key 
  int index = hashFunction(key); 
  
  // find the key in (inex)th list 
  std::list <int> :: iterator i; 
  for (i = table[index].begin(); 
           i != table[index].end(); i++) { 
    if (*i == key) 
      break; 
  } 
  
  // if key is found in hash table, remove it 
  if (i != table[index].end()) 
    table[index].erase(i); 
} 
  
// function to display hash table 
__device__ void Hash::displayHash() { 
  for (int i = 0; i < BUCKET; i++) { 
    cout << i; 
    for (auto x : table[i]) 
      std::cout << " --> " << x; 
    std::cout << endl; 
  } 
} 