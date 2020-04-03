#include "VTK.cuh"
#include "utilities.cuh"

char* VTU_Writer(char path[], int iteration, vec3d* points, int numberOfPoints, float** pointData[], vec3d** vectorData[], std::string pointDataNames[], std::string vectorDataNames[], int size_pointData, int size_vectorData, char* fullpath, int type)
{
	if (type == 0) {
		char buffer[33];
		itoa(iteration, buffer, 10);
		strcpy(fullpath, path);
		strcat(fullpath, "/iter");
		strcat(fullpath, buffer);
		strcat(fullpath, ".vtu");
	}
	else if (type == 1) {
		strcpy(fullpath, path);
		strcat(fullpath, "/boundary.vtu");
	}

	std::ofstream vtu_file;
	vtu_file.open(fullpath);
	//for (int i = 0;i < points.size(); i++)
	vtu_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n"
		<< "<UnstructuredGrid>\n"
		<< "<Piece NumberOfPoints=\"" << numberOfPoints << "\" NumberOfCells=\"" << numberOfPoints << "\">\n"
		<< "<Points>\n"
		<< "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
	for (int i = 0; i < numberOfPoints; i++) {
		vtu_file << points[i].x << " " << points[i].y << " " << points[i].z << "\n";
	}
	vtu_file << "</DataArray>\n"
		<< "</Points>\n";

	vtu_file << "<PointData Scalars=\"density\">\n";

	for (int i = 0; i < size_pointData; i++) {
		float* data = *pointData[i];
		std::string name = pointDataNames[i];

		vtu_file << "<DataArray type=\"Float32\" Name=\"" << name << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
		for (int j = 0; j < numberOfPoints; j++) {
			vtu_file << data[j] << "\n";
		}
		vtu_file << "</DataArray>\n";
	}

	for (int i = 0; i < size_vectorData; i++) {
		vec3d* data = *vectorData[i];
		std::string name = vectorDataNames[i];

		vtu_file << "<DataArray type=\"Float32\" Name=\"" << name << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
		for (int j = 0; j < numberOfPoints; j++) {
			vtu_file << data[j].x << " " << data[j].y << " " << data[j].z << "\n";
		}
		vtu_file << "</DataArray>\n";
	}

	vtu_file << "</PointData>\n"
		<< "<Cells>\n"
		<< "<DataArray type=\"Float32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">\n";

	for (int i = 0; i < numberOfPoints; i++) {
		vtu_file << i << std::endl;
	}

	vtu_file << "</DataArray>\n"
		<< "<DataArray type=\"Float32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">\n";
	for (int i = 0; i < numberOfPoints; i++) {
		vtu_file << i << " ";
	}

	vtu_file << "\n" << "</DataArray>\n"
		<< "<DataArray type=\"Float32\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">\n";

	for (int i = 0; i < numberOfPoints; i++) {
		vtu_file << 1 << " ";
	}

	vtu_file << "\n" << "</DataArray>\n"
		<< "</Cells>\n";

	vtu_file << "</Piece>\n"
		<< "</UnstructuredGrid>\n"
		<< "</VTKFile>";
	vtu_file.close();
	return fullpath;
}

void VTK_Group(char vtk_group_path[], char vtu_path[], float time) {
	//std::cout << vtk_group_path << std::endl;
	char buffer[65];
	strcpy(buffer, clearAddressArray(buffer, vtk_group_path, vtu_path));

	if (fileExists(vtk_group_path) == 0) {
		std::fstream vtk_group;
		vtk_group.open(vtk_group_path);
		vtk_group.seekg(-25, std::ios::end);
		vtk_group << "<DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" file=\"" << buffer << "\"/>\n"
			<< "</Collection>\n"
			<< "</VTKFile>";
		vtk_group.close();
	}
	else {
		std::ofstream vtk_group;
		vtk_group.open(vtk_group_path);
		vtk_group << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
			<< "<Collection>\n"
			<< "<DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" file=\"" << buffer << "\"/>\n"
			<< "</Collection>\n"
			<< "</VTKFile>";
	}

	return;
}