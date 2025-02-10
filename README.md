**Accelerated Computing Documentation**
============================

## Creating an Account

### *NOTE*: Creating an Account Requires an Already Given Username and Password By Your Instructor:

-For *MAC* Users:

- Open the Terminal App
- Run the following, then Press `Enter`:

```bash
ssh YourUsername@dgx.sdsu.edu
```

- Then, It Will Require You to Enter Your Password
  
```
YourUsername@dgx.sdsu.edu's password:
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

* To See Your Folders/Files, Run the Following:
  ```
  ls
  ```

* To Enter into a Specific Folder, Run the Following:
  ```
  cd foldername/
  ```
  
* To Exit a Folder or Go Back to the Previous Folder, Run the Following:
```
cd ..
```

## Creating/Accessing New/Existing Folders and Files

* To Create a New Folder, Run the Following:

```
mkdir foldername
```

* To Create/Access a C (.c) File, Run the Following:
```
emacs -nw filename.c
```

*NOTE*: When You Press `Enter`, It Will Take You Directly To the File.

* To Exit the File and Go Back to the Main Terminal, Press the following:

```
Ctrl+Z
```

* To Go Back to the (.c) File, Run the Following:

```
fg
```

* To Save the File After Writing Some Code, Press the following:
```
Ctrl+X
Ctrl+S
```

## Removing/Deleting Existing Folders and Files

* To Delete an Existing File, Run the following:
```
rm filename
```

* To Remove a File's Content, Run the following:
```
> filename
```
