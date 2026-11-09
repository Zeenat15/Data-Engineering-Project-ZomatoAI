select 
f.restaurant_id, r.restaurant_name, r.city, r.cuisine, 
count(*) as orders,
sum(iff(f.is_delivered, f.sales_amount, 0)) as revenue, 
round(avg(f.customer_rating),2) as avg_customer_rating,
round(avg(f.delivery_time_min),1) as avg_delivery_min
from ZOMATO.marts.fct_orders f 
left join ZOMATO.marts.dim_restaurants r using (restaurant_id) 
group by 1,2,3,4