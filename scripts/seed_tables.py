# scripts/seed_tables.py
import uuid
import random
import datetime
from faker import Faker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID

# Import your models
from app.db.base import Base
from app.models.merchant import Merchant, KYCStatus
from app.models.store import Store
from app.models.product import Product
from app.models.inventory import Inventory
from app.models.order import Order, OrderStatus
from app.models.payment import Payment
from app.models.delivery import Delivery, DeliveryStatus
from app.models.review import Review
from app.models.analytics import Analytics

fake = Faker()

DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/storefront"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def seed_merchants(n=50):
    merchants = []
    for _ in range(n):
        merchant = Merchant(
            merchant_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            wallet_id=uuid.uuid4(),
            kyc_status=random.choice([status.value for status in KYCStatus]),
            created_at=fake.date_time_this_year(),
            updated_at=fake.date_time_this_year()
        )
        merchants.append(merchant)
        session.add(merchant)
    session.commit()
    return merchants

def seed_stores(merchants, n_per_merchant=2):
    stores = []
    for merchant in merchants:
        for _ in range(n_per_merchant):
            store = Store(
                store_id=uuid.uuid4(),
                merchant_id=merchant.merchant_id,
                name=fake.company(),
                description=fake.text(max_nb_chars=200),
                logo_url=fake.image_url(),
                status=random.choice(["active", "suspended", "closed"]),
                created_at=fake.date_time_this_year(),
                updated_at=fake.date_time_this_year()
            )
            stores.append(store)
            session.add(store)
    session.commit()
    return stores

def seed_products(stores, n_per_store=10):
    products = []
    categories = [uuid.uuid4() for _ in range(10)]
    for store in stores:
        for _ in range(n_per_store):
            product = Product(
                product_id=uuid.uuid4(),
                store_id=store.store_id,
                category_id=random.choice(categories),
                name=fake.word().capitalize(),
                description=fake.text(max_nb_chars=150),
                price=round(random.uniform(100, 10000), 2),
                currency="KES",
                status=random.choice(["active", "inactive", "deleted"]),
                created_at=fake.date_time_this_year(),
                updated_at=fake.date_time_this_year()
            )
            products.append(product)
            session.add(product)
    session.commit()
    return products

def seed_inventory(products):
    inventories = []
    for product in products:
        inventory = Inventory(
            inventory_id=uuid.uuid4(),
            product_id=product.product_id,
            stock_quantity=random.randint(0, 100),
            reserved_quantity=random.randint(0, 20),
            updated_at=fake.date_time_this_year()
        )
        inventories.append(inventory)
        session.add(inventory)
    session.commit()
    return inventories

def seed_orders(stores, n_orders=200):
    orders = []
    for _ in range(n_orders):
        store = random.choice(stores)
        order = Order(
            order_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            store_id=store.store_id,
            total_amount=round(random.uniform(500, 50000), 2),
            currency="KES",
            status=random.choice(list(OrderStatus)),
            escrow_id=uuid.uuid4(),
            created_at=fake.date_time_this_year(),
            updated_at=fake.date_time_this_year()
        )
        orders.append(order)
        session.add(order)
    session.commit()
    return orders

def seed_payments(orders):
    payments = []
    for order in orders:
        payment = Payment(
            payment_id=uuid.uuid4(),
            order_id=order.order_id,
            wallet_id=uuid.uuid4(),
            amount=order.total_amount,
            currency=order.currency,
            status=random.choice(["pending", "completed", "failed", "refunded"]),
            created_at=fake.date_time_this_year()
        )
        payments.append(payment)
        session.add(payment)
    session.commit()
    return payments

def seed_deliveries(orders):
    deliveries = []
    for order in orders:
        delivery = Delivery(
            delivery_id=uuid.uuid4(),
            order_id=order.order_id,
            courier_id=uuid.uuid4(),
            status=random.choice([status.value for status in DeliveryStatus]),
            tracking_code=fake.uuid4(),
            created_at=fake.date_time_this_year(),
            updated_at=fake.date_time_this_year()
        )
        deliveries.append(delivery)
        session.add(delivery)
    session.commit()
    return deliveries

def seed_reviews(orders, products):
    reviews = []
    for _ in range(len(orders)//2):
        order = random.choice(orders)
        product = random.choice(products)
        review = Review(
            review_id=uuid.uuid4(),
            order_id=order.order_id,
            product_id=product.product_id,
            customer_id=order.customer_id,
            rating=random.randint(1, 5),
            comment=fake.text(max_nb_chars=100),
            created_at=fake.date_time_this_year()
        )
        reviews.append(review)
        session.add(review)
    session.commit()
    return reviews

def seed_analytics(merchants):
    analytics = []
    for merchant in merchants:
        for period in ["daily", "weekly", "monthly"]:
            report = Analytics(
                report_id=uuid.uuid4(),
                merchant_id=merchant.merchant_id,
                period=period,
                total_sales=round(random.uniform(1000, 100000), 2),
                total_orders=random.randint(1, 500),
                generated_at=fake.date_time_this_year()
            )
            analytics.append(report)
            session.add(report)
    session.commit()
    return analytics

if __name__ == "__main__":
    print("Seeding database...")
    merchants = seed_merchants()
    stores = seed_stores(merchants)
    products = seed_products(stores)
    inventories = seed_inventory(products)
    orders = seed_orders(stores)
    payments = seed_payments(orders)
    deliveries = seed_deliveries(orders)
    reviews = seed_reviews(orders, products)
    analytics = seed_analytics(merchants)
    print("Database seeding complete!")
