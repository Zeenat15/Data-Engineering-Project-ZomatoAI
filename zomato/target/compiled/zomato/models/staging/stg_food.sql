select 
f_id, item as food_name, 
initcap(veg_or_non_veg) as veg_or_non_veg
from zomato.RAW.food where f_id is not null