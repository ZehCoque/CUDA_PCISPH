//defining 3 prime numbers
#include<bits/stdc++.h> 
  
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
    __device__ int hashFunction(vec3d point) { 

        vec3d hashed_point;
        hashed_point.x = p1*point.x;
        hashed_point.y = p2*point.y;
        hashed_point.z = p1*point.z;

        return ((hashed_point.x ^ hashed_point.y ^ hashed_point.z) % BUCKET); 
    } 
  
    __device__ void displayHash(); 
}; 
  
Hash::Hash(int b) 
{ 
    this->BUCKET = b; 
    table = new list<int>[BUCKET]; 
} 
  
void Hash::insertItem(vec3d point) 
{ 
    int index = hashFunction(point); 
    table[index].push_back(key);  
} 
  
void Hash::deleteItem(int key) 
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
void Hash::displayHash() { 
  for (int i = 0; i < BUCKET; i++) { 
    cout << i; 
    for (auto x : table[i]) 
      cout << " --> " << x; 
    cout << endl; 
  } 
} 