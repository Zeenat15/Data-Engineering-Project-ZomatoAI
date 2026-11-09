
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select restaurant_id
from ZOMATO.staging.stg_restaurants
where restaurant_id is null



  
  
      
    ) dbt_internal_test