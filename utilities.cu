#include <windows.h>
#include <cctype>
#include <iostream>
#include <cstring>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <dirent.h>
#include <sstream> 
#include <vector>

struct stat info;

int dirExists(const char* const path)
{
    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? 1 : 0;
}

void CreateDir(char* path) 
{   
    char copy_path[80];
    strcpy(copy_path,path);
    if (dirExists(path) == 1)
    {
        return;
    }
    char chars_array[80];
    strcpy(chars_array, strtok(copy_path, "/"));

    while(chars_array)
    {
        CreateDirectory(chars_array ,NULL);
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
    FILE *input = fopen(path, "r");
    int lineno = 0;
    while (char *line = fgets(linebuf, 1024, input))
    {
        ++lineno;
    }
    fclose(input);
    return lineno;
}

void eraseSubStr(std::string & mainStr, const std::string & toErase)
{
	// Search for the substring in string
	size_t pos = mainStr.find(toErase);
 
	if (pos != std::string::npos)
	{
		// If found then erase it from string
		mainStr.erase(pos, toErase.length());
	}
}

int extractIntegers(char* str){
    char buffer[1024];
    int count = 0;
    for (int i=0;i<strlen(str);i++){
        if (isdigit(str[i])){
            // strcat(buffer,atoi(str[i]));
            buffer[count] = str[i];
            count++;
        }
    }

    return atoi(buffer);
}

char* getMainPath(char *main_path){
    
    if (dirExists("results") == 0){
        strcpy(main_path, "results/simulation 1");
        return main_path;
    }

    strcpy(main_path, "results/simulation ");

    const char* PATH = "./results";

    DIR *dir = opendir(PATH);

    struct dirent *entry = readdir(dir);

    char tmp1[300];
    char tmp2[300];
    std::vector< int > arr;
    strcpy(tmp1, "simulation");

    while (entry != NULL)
    {
        strcpy(tmp2,entry->d_name);
        if (entry->d_type == DT_DIR && strstr(tmp2,tmp1) != 0){
            //printf("%s\n", entry->d_name);
            int integer = extractIntegers(tmp2);
            arr.push_back(integer);
        }
        entry = readdir(dir);
    }

    closedir(dir);

    if (arr.empty()){
        strcat(main_path, "1 ");
        return main_path;
    }

    std::vector<int>::iterator max_value = std::max_element( arr.begin(), arr.end() );

    char buffer [33];
    itoa(max_value[0] + 1,buffer,10);
    strcat(main_path, buffer);

    return main_path;
}