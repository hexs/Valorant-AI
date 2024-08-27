#include <Keyboard.h>
#include <Mouse.h>
#define BUF_SIZE 50

int vel_ = 100;
int delay_ = 0;
char buff[BUF_SIZE + 1];  // +1 for null terminator

//void buff_add(const char* string) {
//  int string_len = strlen(string);
//  int buff_len = strlen(buff);
//
//  if (string_len >= BUF_SIZE) {
//    // If the new string is longer than or equal to the buffer size,
//    // just copy the last BUF_SIZE characters
//    strncpy(buff, string + (string_len - BUF_SIZE), BUF_SIZE);
//  } else if (buff_len + string_len > BUF_SIZE) {
//    // If adding the new string would exceed the buffer size,
//    // shift the existing content and append the new string
//    int shift = buff_len + string_len - BUF_SIZE;
//    memmove(buff, buff + shift, buff_len - shift);
//    strncpy(buff + (BUF_SIZE - string_len), string, string_len);
//  } else {
//    // If there's enough space, simply append the new string
//    strcat(buff, string);
//  }
//
//  buff[BUF_SIZE] = '\0';  // Ensure null termination
//}
void buff_add(const char* string) {
  char cleaned[strlen(string) + 1];
  int j = 0;

  // Remove '\n' and '\r' characters
  for (int i = 0; string[i] != '\0'; i++) {
    if (string[i] != '\n' && string[i] != '\r') {
      cleaned[j++] = string[i];
    }
  }
  cleaned[j] = '\0';

  int cleaned_len = strlen(cleaned);
  int buff_len = strlen(buff);

  if (cleaned_len >= BUF_SIZE) {
    // If the new string is longer than or equal to the buffer size,
    // just copy the last BUF_SIZE characters
    strncpy(buff, cleaned + (cleaned_len - BUF_SIZE), BUF_SIZE);
  } else if (buff_len + cleaned_len > BUF_SIZE) {
    // If adding the new string would exceed the buffer size,
    // shift the existing content and append the new string
    int shift = buff_len + cleaned_len - BUF_SIZE;
    memmove(buff, buff + shift, buff_len - shift);
    strncpy(buff + (BUF_SIZE - cleaned_len), cleaned, cleaned_len);
  } else {
    // If there's enough space, simply append the new string
    strcat(buff, cleaned);
  }

  buff[BUF_SIZE] = '\0';  // Ensure null termination
}

char* check_command(const char* string) {
  int len = strlen(string);
  if (string[len - 1] != '>') {
    return NULL;
  }
  const char* start = strrchr(string, '<');
  if (start != NULL && start < string + len - 1) {
    int command_len = string + len - 1 - start - 1;
    char* command = (char*)malloc(command_len + 1);
    strncpy(command, start + 1, command_len);
    command[command_len] = '\0';
    return command;
  }
  return NULL;
}

void split_function(const char* command, char** result, int* result_size) {
  char temp[strlen(command) + 1];
  strcpy(temp, command);

  *result_size = 0;
  char* token = strtok(temp, "(,)");
  while (token != NULL && *result_size < 10) {  // Limit to 10 parts to avoid overflow
    result[*result_size] = (char*)malloc(strlen(token) + 1);
    strcpy(result[*result_size], token);
    (*result_size)++;
    token = strtok(NULL, "(,)");
  }
}

void move_mouse(int x, int y) {
  while (x != 0 || y != 0) {
    int move_x = (x > vel_) ? vel_ : (x < -vel_) ? -vel_ : x;
    int move_y = (y > vel_) ? vel_ : (y < -vel_) ? -vel_ : y;

    Mouse.move(move_x, move_y);
    x -= move_x;
    y -= move_y;

    delay(delay_);
  }
}

void setup() {
  Serial.begin(9600);
  Keyboard.begin();
  Mouse.begin();

  delay(1000);
  Serial.println("+++");
  // Initialize buffer with dashes
  memset(buff, '-', BUF_SIZE);
  buff[BUF_SIZE] = '\0';
}

void loop() {
  while (1) {
    if (Serial.available()) {
      // Read new character from Serial
      char ch = Serial.read();
      char str[2];
      str[0] = ch;
      str[1] = '\0';
      buff_add(str);
      //      Serial.println(buff);
      if (ch == '>')
        break;
    }
  }
  Serial.println("!!!!!!!!!!!!");
  Serial.println(buff);
  char* command = check_command(buff);
  Serial.println(command);

  char* split_result[10];  // Assuming max 10 parts
  int split_size;
  split_function(command, split_result, &split_size);

  //  Serial.print("Split result: ");
  //  for (int i = 0; i < split_size; i++) {
  //    Serial.print(split_result[i]);
  //    if (i < split_size - 1) Serial.print(", ");
  //  }
  //  Serial.println();

  if (strcmp(split_result[0], "move") == 0) {
    int x = atoi(split_result[1]);
    int y = atoi(split_result[2]);

    // Move mouse by x, y coordinates
    move_mouse(x, y);
  }
  Serial.println("############");
}
