#include <Keyboard.h>
#include <Mouse.h>

const int BUF_SIZE = 10;
char buf[BUF_SIZE] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
int vel_ = 100;
int delay_ = 0;

void move_mouse(int x, int y);
void setup() {
  Serial.begin(921600);
  Keyboard.begin();
  Mouse.begin();
}
void loop() {
  if (Serial.available()) {
    // Read new character from Serial
    char ch = Serial.read();

    // Shift buffer to the left by one
    for (int i = 0; i < BUF_SIZE - 1; i++) {
      buf[i] = buf[i + 1];
    }
    // Add new character to the end of the buffer
    buf[BUF_SIZE - 1] = ch;

    //                               0123456789
    // Check for mouse move command: <+xxx-yyy>
    // x and y == 0 to 999
    // vel_   == 005 to 120
    // delay_ == 000 to 001
    if (buf[0] == '<' && buf[9] == '>') {
      // Extract x and y coordinates as strings
      char x_str[5] = { buf[1], buf[2], buf[3], buf[4], '\0' };
      char y_str[5] = { buf[5], buf[6], buf[7], buf[8], '\0' };
      // Convert strings to integers
      int x = atoi(x_str);
      int y = atoi(y_str);
      // Move mouse by x, y coordinates
      move_mouse(x, y);
    }

    //                                0123456789
    // Check for mouse click command: -------<c>
    if (buf[7] == '<' && buf[9] == '>') {
      if (buf[8] == 'c') {
        Mouse.click();
      }
    }

    //                                     0123456789
    // Check for velocity setting command: --<velxxx>
    if (buf[2] == '<' && buf[9] == '>') {
      if (buf[3] == 'v' && buf[4] == 'e' && buf[5] == 'l') {
        char str[4] = { buf[6], buf[7], buf[8], '\0' };
        vel_ = atoi(str);
        if (vel_ > 127)
          vel_ = 127;
      }
    }

    //                                  0123456789
    // Check for delay setting command: <delayxxx>
    if (buf[0] == '<' && buf[9] == '>') {
      if (buf[1] == 'd' && buf[2] == 'e' && buf[3] == 'l' && buf[4] == 'a' && buf[5] == 'y') {
        char str[4] = { buf[6], buf[7], buf[8], '\0' };
        delay_ = atoi(str);
      }
    }

    //                        0123456789
    // Check for get command: -----<get>
    char target[] = "<get>";
    if (strncmp(buf + 5, "<get>", 5) == 0) {
      Serial.print("vel_ ");
      Serial.println(vel_);
      Serial.print("delay_ ");
      Serial.println(delay_);
    }
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