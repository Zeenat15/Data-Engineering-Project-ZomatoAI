
  
    

        create or replace transient table ZOMATO.marts.dim_food
         as
        (select 
    f_id, 
    food_name, 
    veg_or_non_veg 
from ZOMATO.staging.stg_food
        );
      
  