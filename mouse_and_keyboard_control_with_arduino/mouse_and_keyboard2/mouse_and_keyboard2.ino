#include <Keyboard.h>
#include <Mouse.h>

int x;

void setup() {
  Serial.begin(9600);

  Keyboard.begin();
  Mouse.begin();
}

void loop() {
  if (Serial.available()) {
    x = Serial.read();
    // Serial.println(x);
    if (x == 'w') Mouse.move(0, -3);
    if (x == 'a') Mouse.move(-3, 0);
    if (x == 's') Mouse.move(0, 3);
    if (x == 'd') Mouse.move(3, 0);

    if (x == 'W') Mouse.move(0, -30);
    if (x == 'A') Mouse.move(-30, 0);
    if (x == 'S') Mouse.move(0, 30);
    if (x == 'D') Mouse.move(30, 0);


    if (x == 'c') Mouse.click();
    if (x == 'p') Mouse.press();
    if (x == 'r') Mouse.release();
    delayMicroseconds(1);
  }
}
