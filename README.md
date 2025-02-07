**Accelerated Computing Documentation**
============================

## Creating an Account

### *NOTE*: Creating an Account Requires an Already Given Username and Password By Your Instructor:

-For *MAC* Users:

- Open the Terminal App
- Enter the Following, then Press `Enter`:

```bash
ssh <YourUsername>@dgx.sdsu.edu
```

- Then, It Will Require You to Enter Your Password
  
```
<YourUsername>@dgx.sdsu.edu's password:
```

*NOTE*: DON'T ENTER YOUR PASSWORD MANUALLY, COPY AND PASTE IT INTO YOUR TERMINAL

* If the Password You Entered is Wrong, You Will Get the Following Message:

```
  Permission denied, please try again.
```

* If the Password You Entered is Correct, You Will See the Following Message as a Successful Login:

```
Welcome to NVIDIA DGX Server Version #.#.# (GNU/Linux #.#.#-###-generic x##_##)
```

## Viewing/Switching Between Folders and Files

* To See Your Folders/Files, Enter the Following:
  ```
  ls
  ```

* To Enter into a Specific Folder, Enter the Following:
  ```
  cd foldername/
  ```

## Creating New Folders and Files

* To Create a New Folder, Enter the Following:

```
mkdir foldername
```

* To Create a C (.c) File, Enter the Following:
```
emacs -nw filename.c
```

*NOTE*: When You Press `Enter`, It Will Take You Directly To the File.

* To Exit the File and Go Back to the Main Terminal, Press the Following:

```
Ctrl+Z
```

* To Go Back to the (.c) File, Enter the Following:

```
fg
```

* To Save the File After Writing Some Code, Press the Following:
```
Ctrl+X
Ctrl+S
```
