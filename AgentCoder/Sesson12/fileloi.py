# ecommerce_analytics.py
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Product:
    product_id: str
    name: str
    price: float
    category: str

@dataclass
class OrderItem:
    product: Product
    quantity: int

@dataclass
class Order:
    order_id: str
    customer_id: str
    items: List[OrderItem]
    order_date: datetime
    coupon_code: Optional[str] = None

@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    join_date: datetime


class OrderProcessor:
    def __init__(self, tax_rate: float = 0.08):
        self.tax_rate = tax_rate
        self.coupon_registry: Dict[str, float] = {
            "SUMMER20": 20.0,  # Giảm 20%
            "WELCOME10": 10.0, # Giảm 10%
            "VIP50": 50.0      # Giảm 50%
        }

    def calculate_subtotal(self, order: Order) -> float:
        """Tính tổng tiền hàng trước thuế và giảm giá."""
        return sum(item.product.price * item.quantity for item in order.items)

    def apply_coupon(self, order: Order, subtotal: float) -> float:
        """
        Tính số tiền được giảm dựa trên mã coupon.
        """
        if not order.coupon_code:
            return 0.0
            
        coupon_upper = order.coupon_code.upper()
        if coupon_upper in self.coupon_registry:
            coupon_rate = self.coupon_registry[coupon_upper]
            # -------------------------------------------------------------
            # ĐÃ SỬA LỖI SỐ 1 (NameError): Đổi 'promo_rate' thành 'coupon_rate'
            # -------------------------------------------------------------
            discount_amount = subtotal * (coupon_rate / 100.0)
            return round(discount_amount, 2)
            
        return 0.0

    def process_order_total(self, order: Order) -> float:
        """Tính tổng số tiền cuối cùng của đơn hàng (Đã áp thuế và coupon)."""
        subtotal = self.calculate_subtotal(order)
        discount = self.apply_coupon(order, subtotal)
        
        taxable_amount = max(0.0, subtotal - discount)
        tax = taxable_amount * self.tax_rate
        
        return round(taxable_amount + tax, 2)


class AnalyticsEngine:
    def __init__(self):
        pass

    def calculate_average_order_value(self, customer_id: str, orders: List[Order], processor: OrderProcessor) -> float:
        """Tính toán giá trị đơn hàng trung bình (AOV) của một khách hàng."""
        customer_orders = [o for o in orders if o.customer_id == customer_id]
        if not customer_orders:
            return 0.0
            
        total_spent = sum(processor.process_order_total(o) for o in customer_orders)
        return round(total_spent / len(customer_orders), 2)

    def calculate_customer_lifetime_value(self, customer_id: str, orders: List[Order], processor: OrderProcessor) -> float:
        """
        Tính toán Giá trị Vòng đời Khách hàng (CLV - Customer Lifetime Value).
        Công thức: CLV = (AOV * Tần suất mua hàng trung bình hàng tháng) * Thời gian gắn bó giả định (tháng).
        """
        customer_orders = [o for o in orders if o.customer_id == customer_id]
        
        # Kiểm tra nếu khách hàng không có đơn hàng nào
        if not customer_orders:
            return 0.0
            
        # Giả định thời gian gắn bó mặc định là 12 tháng
        assumed_lifespan_months = 12
        
        # Tính toán tổng chi tiêu của khách hàng
        total_spent = sum(processor.process_order_total(o) for o in customer_orders)
        
        # -------------------------------------------------------------
        # ĐÃ SỬA LỖI SỐ 2 (ZeroDivisionError): Thêm kiểm tra biên
        # -------------------------------------------------------------
        purchase_frequency_per_month = len(customer_orders) / 6.0 # Giả định tần suất trong nửa năm (6 tháng)
        average_order_value = total_spent / len(customer_orders)
        
        clv = (average_order_value * purchase_frequency_per_month) * assumed_lifespan_months
        return round(clv, 2)

    def generate_sales_report(self, orders: List[Order], processor: OrderProcessor) -> Dict[str, float]:
        """Tạo báo cáo doanh thu tổng hợp theo danh mục sản phẩm."""
        report: Dict[str, float] = {}
        for order in orders:
            for item in order.items:
                category = item.product.category
                item_total = item.product.price * item.quantity
                report[category] = report.get(category, 0.0) + item_total
        return report