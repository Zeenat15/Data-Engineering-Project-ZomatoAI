
select oi.order_item_id, oi.order_id, oi.restaurant_id, oi.f_id, o.order_timestamp as order_ts,
       o.order_date, o.city, oi.price, oi.quantity, oi.line_amount
from ZOMATO.staging.stg_order_items oi
inner join ZOMATO.staging.stg_orders o using (order_id)

  where o.order_timestamp > (select coalesce(max(order_ts),'1900-01-01'::timestamp) from ZOMATO.marts.fct_order_items)
