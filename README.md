# APS

## Table of contents

+ [General info](#general-info)
+ [Requirements](#requirements)
+ [Additional Program Requirement](#additional-program-requirement)
+ [How to Use](#how-to-use)



## General info

The program __APS__ is the __Reinforcement Learning based Model for Gantt Planning.__

__Time window scanner is used for solving scheduling problems.__

##  Requirements

This module requires the following modules:

+ python=3.5(3.5.6)
+ tensorflow-gpu==1.14.0 (install gpu version of tensorflow module)
+ tensorflow==1.14.0  (install cpu version of tensorflow module)
+ scipy==1.2.1
+ pygame
+ moviepy
+ numpy==1.18.5
+ pandas
+ matplotlib
+ PyCharm under version of 2020.1



## Additional Program Requirement

In order to generate __gif__ file, __ImageMagik__ program is also required.

[ImageMagik](https://www.imagemagick.org/script/index.php)

If you are using __window OS__ , 

you should do some additional works follows...

In __config_defaults.py__ , which has a directory :  

C:\Users\user\Anaconda3\envs\\'virtual_env_name'\Lib\site-packages\moviepy , change the code

```python
IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'auto-detect')
```

into

```python
IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.9-Q16\magick.exe')
```

`ImageMagick-7.0.9-Q16` is the ImageMagik version you had installed.

----------







## How to Use



### Setting target date or projects



You can either import schedule by __target date__ or __target projects__



+ __Setting target date__

  If you want to import training/test data by target date, you can simply send __target_start_date__ and __target_finish_date__ to __work.py__ > <function>__import_schedule_by_target_date(filepath, target_start_date, target_finish_date)__

  

   ```python
    import_schedule_by_target_date('../environment/data/191227_납기일 추가.xlsx', '20190201', '20190601')
   ```



+ __Setting target projects__

  If you want to import training/test data by target projects, you can simply send __target_projects__ to __work.py__ > <function>__import_schedule_by_target_projects(filepath, target_projects)__

  

  ```python
      import_schedule_by_target_projects('../environment/data/191227_납기일 추가.xlsx', [3095, 'R873'])
  ```



-----------------------------------------------





+ #### Learning Parameters Configurations

  + __Time Window Size__

    In __train.py__ > __main__ function

    ```python
    window = (10, 40)
    ```

  ​       You can determine the size of __Time Window__ which scans the total imported schedule

  

  + __Learning Rate__

    In __train.py__ > __main__ function

    ```python
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5)
    ```

    A recommanded figure for learning rate is __1e-5__ 

    

  + __Discount Rate__

    In __train.py__ > __main__ function

    ```python
    gamma = .99 # discount rate for advantage estimation and reward discounting
    ```

    A recommanded figure for discount rate is __0.99__

    

  + __Frequency of target model update(N-step bootstrapping)__

    In __A3C__ Algorithm, _each of_ __Workers__ collects samples  with its own environments.  After a __certain number of time-steps__ , target network(global network) is updated with that samples.

    In __train.py__ > <class>__Worker__><member function>__work__

    ```python
    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length-1: # in this case, frequency of target model update is 30 time-steps
    ```

    

  + Number of Threads

    You can also set up number of threads by changing the number of __Workers__

    In __train.py__ > __main__ function

    ```python
    num_workers = nultiprocessing.cpu.count() # Set workers to number of available CPU threads
    if num_workers > 8:
        num_workers = 8 
    workers = []
    ```

  
