// basic file operations
#include <iostream>
#include <fstream>
#include <string>

int VTU_Writer(std::string path,int iteration,vec3d* points,int numberOfPoints, float** pointData[],vec3d** vectorData[], std::string pointDataNames[], std::string vectorDataNames[], int size_pointData, int size_vectorData) 
{
  path = path + "iter" + std::to_string(iteration) + ".vtu";
  std::ofstream vtu_file;
  vtu_file.open (path);
  //for (int i = 0;i < points.size(); i++)
  vtu_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n"
           << "<UnstructuredGrid>\n" 
           << "<Piece NumberOfPoints=\"" << numberOfPoints << "\" NumberOfCells=\"" << numberOfPoints << "\">\n"
           << "<Points>\n"
           << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0;i < numberOfPoints; i++)  {
    vtu_file << points[i].x << " " << points[i].y << " " <<  points[i].z << "\n";
  }
  vtu_file  << "</DataArray>\n" 
           << "</Points>\n" ;

  vtu_file << "<PointData Scalars=\"density\">\n";

  for (int i = 0; i < size_pointData ;i++){

    float* data = *pointData[i];
    std::string name = pointDataNames[i];

    vtu_file << "<DataArray type=\"Float32\" Name=\""<< name << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int j = 0; j < numberOfPoints; j++)  {
        vtu_file << data[j] << "\n";
    }
    vtu_file  << "</DataArray>\n";
  }

  for (int i = 0; i < size_vectorData; i++){

    vec3d* data = *vectorData[i];
    std::string name = vectorDataNames[i];

    vtu_file << "<DataArray type=\"Float32\" Name=\""<< name << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int j = 0; j < numberOfPoints; j++)  {
        vtu_file << data[j].x << " " << data[j].y << " " << data[j].z << "\n";
    }
    vtu_file  << "</DataArray>\n";
  }

  vtu_file << "</PointData>\n" 
           << "<Cells>\n" 
           << "<DataArray type=\"Float32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">\n";

  for (int i = 0;i < numberOfPoints; i++){
    vtu_file << i << std::endl;
  }
  
  vtu_file  << "</DataArray>\n" 
           << "<DataArray type=\"Float32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for (int i = 0;i < numberOfPoints; i++){
    vtu_file << i << " ";
  }         

  vtu_file << "\n"  << "</DataArray>\n" 
           << "<DataArray type=\"Float32\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">\n" ;

  for (int i = 0;i < numberOfPoints; i++){
    vtu_file << 1 << " ";
  }     

  vtu_file << "\n"  << "</DataArray>\n" 
           << "</Cells>\n" ;

  vtu_file << "</Piece>\n" 
           << "</UnstructuredGrid>\n" 
           << "</VTKFile>";
  vtu_file.close();
  return 0;
}