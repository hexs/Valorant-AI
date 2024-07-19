#include <Keyboard.h>
#include <Mouse.h>

void setup() {
  Serial.begin(9600);
  Keyboard.begin();
  Mouse.begin();
}

void loop() {
  if (Serial.available()) {
    String str = Serial.readString();
    Serial.print(str);
    
    char ac = str[0];
    char char1 = str[1];
    char char2 = str[2];
    char char3 = str[3];
    String string_ = str.substring(1);

    // 'a' Keyboard.press(key)
    // key: the key to press. Allowed data types: char.
    if (ac == 'a') Keyboard.press(char1);

    // 'b' Keyboard.print(characters)
    // characters: a string to be sent to the computer as keystrokes.
    if (ac == 'b') Keyboard.print(string_);

    // 'c' Keyboard.println(characters)
    // characters: a string to be sent to the computer as keystrokes, followed by Enter.
    if (ac == 'c') Keyboard.println(string_);

    // 'd' Keyboard.release(key)
    // key: the key to release. Allowed data types: char.
    if (ac == 'd') Keyboard.release(char1);

    // 'e' Keyboard.releaseAll()
    if (ac == 'e') Keyboard.releaseAll();

    // 'f' Keyboard.write(character)
    // character: a char or int
    if (ac == 'f') Keyboard.write(char1);


    // 'g' Mouse.click(button)
    // button: which mouse button to press. Allowed data types: char.
    //    MOUSE_LEFT (default)
    //    MOUSE_RIGHT
    //    MOUSE_MIDDLE
    if (ac == 'g') Mouse.click(char1);


    // 'h' Mouse.move(xVal, yVal, wheel)
    // xVal: amount to move along the x-axis. Allowed data types: signed char.
    // yVal: amount to move along the y-axis. Allowed data types: signed char.
    // wheel: amount to move scroll wheel. Allowed data types: signed char.
    if (ac == 'h') Mouse.move(char1, char2, char3);

    // 'i' Mouse.press(button)
    // button: which mouse button to press. Allowed data types: char.
    //     MOUSE_LEFT (default)
    //     MOUSE_RIGHT
    //     MOUSE_MIDDLE
    if (ac == 'i') Mouse.press(char1);

    // 'j' Mouse.release(button)
    // button: which mouse button to press. Allowed data types: char.
    //     MOUSE_LEFT (default)
    //     MOUSE_RIGHT
    //     MOUSE_MIDDLE
    if (ac == 'j') Mouse.release(char1);

    // 'k' Mouse.isPressed(button);
    // button: which mouse button to check. Allowed data types: char.
    //     MOUSE_LEFT (default)
    //     MOUSE_RIGHT
    //     MOUSE_MIDDLE
    // Returns: Reports whether a button is pressed or not. Data type: bool.



    // #define MOUSE_LEFT 1
    // #define MOUSE_RIGHT 2
    // #define MOUSE_MIDDLE 4
    // #define MOUSE_ALL (MOUSE_LEFT | MOUSE_RIGHT | MOUSE_MIDDLE)
  }
}