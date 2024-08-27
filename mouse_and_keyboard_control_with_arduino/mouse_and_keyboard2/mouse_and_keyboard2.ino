#include <Keyboard.h>
#include <Mouse.h>

#define BUFFER_SIZE 50

int velocity = 100;
int delayTime = 0; // microsecond
char buffer[BUFFER_SIZE + 1];  // +1 for null terminator

void addToBuffer(const char* input) {
  char cleanedInput[strlen(input) + 1];
  int j = 0;

  // Remove '\n' and '\r' characters
  for (int i = 0; input[i] != '\0'; i++) {
    if (input[i] != '\n' && input[i] != '\r') {
      cleanedInput[j++] = input[i];
    }
  }
  cleanedInput[j] = '\0';

  int cleanedLength = strlen(cleanedInput);
  int bufferLength = strlen(buffer);

  if (cleanedLength >= BUFFER_SIZE) {
    // If the new string is longer than or equal to the buffer size,
    // copy the last BUFFER_SIZE characters
    strncpy(buffer, cleanedInput + (cleanedLength - BUFFER_SIZE), BUFFER_SIZE);
  }
  else if (bufferLength + cleanedLength > BUFFER_SIZE) {
    // If adding the new string would exceed the buffer size,
    // shift the existing content and append the new string
    int shift = bufferLength + cleanedLength - BUFFER_SIZE;
    memmove(buffer, buffer + shift, bufferLength - shift);
    strncpy(buffer + (BUFFER_SIZE - cleanedLength), cleanedInput, cleanedLength);
  }
  else {
    // If there's enough space, simply append the new string
    strcat(buffer, cleanedInput);
  }

  buffer[BUFFER_SIZE] = '\0';  // Ensure null termination
}

char* extractCommand(const char* input) {
  int length = strlen(input);
  if (input[length - 1] != '>') {
    return NULL;
  }
  const char* start = strrchr(input, '<');
  if (start < input + length - 1) {
    int commandLength = input + length - 1 - start - 1;
    char* command = (char*)malloc(commandLength + 1);
    strncpy(command, start + 1, commandLength);
    command[commandLength] = '\0';
    return command;
  }
  return NULL;
}

void splitCommand(const char* command, char** result, int* resultSize) {
  char temp[strlen(command) + 1];
  strcpy(temp, command);

  *resultSize = 0;
  char* token = strtok(temp, "(,)");
  while (token != NULL && *resultSize < 10) {  // Limit to 10 parts to avoid overflow
    result[*resultSize] = (char*)malloc(strlen(token) + 1);
    strcpy(result[*resultSize], token);
    (*resultSize)++;
    token = strtok(NULL, "(,)");
  }
}

void moveMouse(int x, int y) {
  while (x != 0 || y != 0) {
    int moveX = (x > velocity) ? velocity : (x < -velocity) ? -velocity : x;
    int moveY = (y > velocity) ? velocity : (y < -velocity) ? -velocity : y;

    Mouse.move(moveX, moveY);
    x -= moveX;
    y -= moveY;

    delay(delayTime);
  }
}

void setup() {
  Serial.begin(9600);
  Keyboard.begin();
  Mouse.begin();

  delay(1000);
  Serial.println("+++");
  // Initialize buffer with dashes
  memset(buffer, '-', BUFFER_SIZE);
  buffer[BUFFER_SIZE] = '\0';
}

void loop() {
  while (1) {
    if (Serial.available()) {
      // Read new character from Serial
      char ch = Serial.read();
      char str[2] = {ch, '\0'};
      addToBuffer(str);
      if (ch == '>') {
        break;
      }
    }
  }
  char* command = extractCommand(buffer);
  //Serial.println("!!!!!!!!!!!!");
  //Serial.println(buffer);
  //Serial.println(command);

  char* splitResult[10];  // Assuming max 10 parts
  int splitSize;
  splitCommand(command, splitResult, &splitSize);

  if (strcmp(splitResult[0], "move") == 0) {
    int x = atoi(splitResult[1]);
    int y = atoi(splitResult[2]);

    // Move mouse by x, y coordinates
    moveMouse(x, y);
  }


  // Free allocated memory for splitResult
  for (int i = 0; i < splitSize; i++) {
    free(splitResult[i]);
  }
  free(command);  // Free the allocated memory for command
}
