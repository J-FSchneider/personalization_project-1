from utils.preprocessing import *

def time_of_day_encoded(data):
    """
    Converts the moment of day buckets to categorical varibales in different columns.
    Input: pandas DataFrame
    Output: Appended dataframe(Use data = time_of_day_categorical(data) to change the
    existing Dataframe)
    
    """
    parse_ts_listen(data) #Preprocess
    parse_moment_of_day(data) #Preprocess
   if "moment_of_day" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'moment_of_day'")
      #Making the dummy variable from the categorical 'moment_of_day'   
    temp1 = pd.get_dummies(data.moment_of_day)
    
      #Joining the new dataset with the encoding to the original data
    data = pd.concat([data,temp1],axis =1) 
    
    del temp1
    
    return data


def age_bucket_encoded(data):

    """
    Converts the age buckets to categorical varibales in different columns.
    Input: pandas DataFrame
    Output: Appended dataframe(Use data = age_categorical(data) to change the
    existing Dataframe)
    
    """
	parse_user_age(data) #preprocess
   
    if "user_age_bucket" not in data:
        raise IOError("The input dataframe does not contain "
                      "the column 'user_age_bucket'")
        
     #Making the dummy variable from the categorical 'user_age_bucket'   
    temp = pd.get_dummies(data.user_age_bucket)
    # Joining the new dataset with the encoding to the original data
    data = pd.concat([data, temp],axis =1) 
    
    del temp
    
    return data

def modified_data(data) :	
    """
    This function calls the functions mentioned above and stores the values in the dataset 
    to output the new dataset with the encoded values. It also deletes the 
    'user_age_bucket' and 'moment_of_day' column as they are not required anymore
    """
  data = time_of_day_encoded(data) #storing new dataset
  data = age_bucket_encoded(data) #storing new dataset
  del data['user_age_bucket'] #removing the 'user_age_bucket' column
  del data['moment_of_day'] #Removing the 'moment_of_day' column
  
  return data
