#include "utilities.cuh"
struct stat info;
int fileExists(const char* const path) {
	return stat(path, &info);
}

int dirExists(const char* const path)
{
	int statRC = stat(path, &info);
	if (statRC != 0)
	{
		if (errno == ENOENT) { return 0; } // something along the path does not exist
		if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
		return -1;
	}

	return (info.st_mode & S_IFDIR) ? 1 : 0;
}

void CreateDir(char* path)
{
	char copy_path[1024];
	strcpy(copy_path, path);

	if (dirExists(path) == 1)
	{
		return;
	}
	char chars_array[1024];
	strcpy(chars_array, strtok(copy_path, "/"));

	while (chars_array)
	{
		CreateDirectory(chars_array, NULL);
		if (dirExists(path) == 1)
		{
			return;
		}
		
		strcat(chars_array, "/");
		strcat(chars_array, strtok(NULL, "/"));
		
		
		
	}
	return;
}

int count_lines(char path[])
{
	char linebuf[1024];
	FILE* input = fopen(path, "r");
	int lineno = 0;
	while (char* line = fgets(linebuf, 1024, input))
	{
		++lineno;
	}
	fclose(input);
	return lineno;
}

int extractIntegers(char* str) {
	char* buffer = new char[strlen(str)];
	int count = 0;
	//printf("%zd\n", strlen(str));
	for (int i = 0; i < strlen(str); i++) {
		if (isdigit(str[i])) {
			// strcat(buffer,atoi(str[i]));
			buffer[count] = str[i];
			count++;
		}
	}
	//printf("buffer = %s\n",buffer);
	return atoi(buffer);
}

char* getMainPath(char* main_path) {

	char* starter_path = new char[256];

	if (strlen(user_results_folder) == 0) {
		starter_path = "./results";
	}
	else {
		starter_path = user_results_folder;
	}

	char* PATH = new char[256];
	strcpy(PATH, starter_path);
	strcat(PATH, "/");

	if (dirExists(PATH) == 0) {
		strcpy(main_path, starter_path);
		strcat(main_path, "/simulation 1");
		return main_path;
	}

	strcpy(main_path, starter_path);
	strcat(main_path, "/simulation ");

	DIR* dir = opendir(PATH);

	struct dirent* entry = readdir(dir);

	char tmp1[1024];
	char tmp2[1024];
	std::vector<int>arr;
	strcpy(tmp1, "simulation");
	int integer;
	while (entry != NULL)
	{
		strcpy(tmp2, entry->d_name);
		// printf("%s\n", entry->d_name);
		if (entry->d_type == DT_DIR && strstr(tmp2, tmp1) != 0) {
			//printf("%s\n", entry->d_name);
			integer = extractIntegers(tmp2);
			//printf("%d\n", integer);
			arr.push_back(integer);
		}
		entry = readdir(dir);
	}

	closedir(dir);

	if (arr.empty()) {
		strcat(main_path, "1");
		return main_path;
	}

	std::vector<int>::iterator max_value = std::max_element(arr.begin(), arr.end());
	

	char* buffer = new char[10];
	itoa(max_value[0] + 1, buffer, 10);

	strcat(main_path, buffer);

	return main_path;
}

char* clearAddressArray(char* buffer, char* s1, char* s2)
{
	size_t size1 = strlen(s1);
	size_t size2 = strlen(s2);
	size_t max_len;

	if (size1 > size2)
	{
		max_len = size1;
		buffer = s1;
	}
	else
	{
		max_len = size2;
		buffer = s2;
	}

	int count = 0;

	for (int i = 0; i < max_len; i++) {
		if (s1[i] == s2[i]) {
			count++;
		}
	}

	for (int i = 0; i < count; i++) {
		for (int k = 0; k < max_len; k++) {
			buffer[k] = buffer[k + 1];
		}
	}

	return buffer;
}

void cudaAtributes(void *dev_ptr) {
	cudaPointerAttributes* atributes = new cudaPointerAttributes;
	cudaPointerGetAttributes(atributes, dev_ptr);
	printf("%d %d %d hptr = %p dptr = %p\n", static_cast<int>(atributes->memoryType), atributes->device, atributes->isManaged, atributes->devicePointer, atributes->hostPointer);
	return;
}

void writeTimeKeeper(char* main_path) {

	char buffer[1024];
	strcpy(buffer, main_path);
	char* time_keeper_path = strcat(buffer, "/time_keeper.dat");

	std::ofstream time_keeper;
	time_keeper.open(time_keeper_path, std::ios::app);

	time_keeper << iteration << " " << simulation_time << "\n";

	time_keeper.close();

}

void getNewSimTime(char* main_path) {

	char buff1[100];
	strcpy(buff1, main_path);
	char* time_keeper_path = strcat(buff1, "/time_keeper.dat");

	std::ifstream time_keeper(time_keeper_path);

	char buff2[100];
	strcpy(buff2, main_path);
	char* time_keeper_path_2 = strcat(buff2, "/time_keeper2.dat");

	std::ofstream time_keeper2;
	time_keeper2.open(time_keeper_path_2);

	time_keeper.seekg(0, std::ios::beg);

	char* num_buffer = new char[50];
	int buff_index = 0;
	int num;
	bool newline = true;
	for (char write2line; time_keeper.get(write2line);) {
		time_keeper2 << write2line;
		if (isdigit(write2line) && newline) {
			num_buffer[buff_index] = write2line;
			buff_index++;
		}
		else if (write2line == 32) {
			num = atoi(num_buffer);
			num_buffer = new char[50];
			buff_index = 0;
			if (num == iteration) {
				break;
			}
			newline = false;
			
		}
		else if (write2line == 10) {
			newline = true;
		}

	}

	for (char write2line; time_keeper.get(write2line);) {
		time_keeper2 << write2line;
		if (isdigit(write2line) || write2line == 46) {
			num_buffer[buff_index] = write2line;
			buff_index++;
		}
		else if (write2line == 10) {
			break;
		}
	}

	simulation_time = (float)atof(num_buffer);
	time_keeper.close();
	time_keeper2.close();
	remove(time_keeper_path);
	rename(time_keeper_path_2, time_keeper_path);

	return;
}

double dround(double val, int dp) {
	int charsNeeded = 1 + snprintf(NULL, 0, "%.*f", dp, val);
	char* buffer = (char*)malloc(charsNeeded);
	snprintf(buffer, charsNeeded, "%.*f", dp, val);
	double result = atof(buffer);
	free(buffer);
	return result;
}

void displayProgress(std::chrono::high_resolution_clock::time_point start) {

	float progress = simulation_time / final_time;

	int barWidth = 15;
	std::cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	
	std::chrono::high_resolution_clock::time_point current_time = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point;
	auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start).count();

	std::cout << "] " << dround(progress * 100.0, 2) << "% | " << 
		"Elapsed time: "<< elapsed_time << " s | Time step: " << dround(delta_t * 1000,3) << " ms | Iteration Number: " << iteration << " | Max Density Error: " << round(max_rho_err) << "    \r";
	std::cout.flush();

	return;
}

void rewritePVD(char* main_path) {

	char* num_buff = new char[32];
	itoa(iteration, num_buff, 10);
	char search_char[64];
	strcpy(search_char, "iter");
	strcat(search_char, num_buff);

	char pvd_path[256];
	strcpy(pvd_path, main_path);
	strcat(pvd_path, "/PCISPH.pvd");

	char pvd_path_cpy[256];
	strcpy(pvd_path_cpy, main_path);
	strcat(pvd_path_cpy, "/PCISPH2.pvd");

	std::ifstream pvd1(pvd_path);

	std::ofstream pvd2;
	pvd2.open(pvd_path_cpy);

	char* row = new char[1024];
	int row_index = 0;
	for (char write2line; pvd1.get(write2line);) {
	
		pvd2 << write2line;
		row[row_index] = write2line;

		if (write2line == 10) {
			if (strstr(row, search_char) != nullptr) {
				break;
			}
			else {
				row = new char[1024];
				row_index = 0;
			}
		}

	}

	pvd2 << "</Collection>\n"
		 << "</VTKFile>";

	pvd1.close();
	pvd2.close();

	remove(pvd_path);
	rename(pvd_path_cpy, pvd_path);

	for (int i = 1; i < 4; i++) {
		num_buff = new char[32];
		itoa(iteration + i, num_buff, 10);

		char* vtu = new char[256];
		strcpy(vtu, main_path);
		strcat(vtu, "/vtu/iter");
		strcat(vtu, num_buff);
		strcat(vtu, ".vtu.");

		if (fileExists(vtu) == 1) {
			remove(vtu);
		}
	}

	return;

}

