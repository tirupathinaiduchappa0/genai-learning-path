"""
Setup script — Creates a sample SQLite database with eCommerce data.
Run this ONCE before running the agent.
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecommerce.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.executescript("""
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS order_items;

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        city TEXT,
        joined_date TEXT
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        stock INTEGER
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date TEXT,
        total_amount REAL,
        status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );

    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );

    -- Insert sample customers
    INSERT INTO customers VALUES (1, 'Tirupathi Naidu', 'tiru@gmail.com', 'Hyderabad', '2024-01-15');
    INSERT INTO customers VALUES (2, 'Priya Sharma', 'priya@gmail.com', 'Mumbai', '2024-02-20');
    INSERT INTO customers VALUES (3, 'Rahul Verma', 'rahul@gmail.com', 'Delhi', '2024-03-10');
    INSERT INTO customers VALUES (4, 'Anita Patel', 'anita@gmail.com', 'Ahmedabad', '2024-04-05');
    INSERT INTO customers VALUES (5, 'Karthik Reddy', 'karthik@gmail.com', 'Bangalore', '2024-05-12');

    -- Insert sample products
    INSERT INTO products VALUES (1, 'Garden Bench', 'Furniture', 89.99, 50);
    INSERT INTO products VALUES (2, 'Rattan Chair Set', 'Furniture', 299.99, 20);
    INSERT INTO products VALUES (3, 'Solar Garden Light', 'Lighting', 24.99, 200);
    INSERT INTO products VALUES (4, 'Wooden Planter Box', 'Garden', 45.99, 100);
    INSERT INTO products VALUES (5, 'Outdoor Parasol', 'Furniture', 129.99, 30);
    INSERT INTO products VALUES (6, 'Garden Hose 30m', 'Tools', 34.99, 150);
    INSERT INTO products VALUES (7, 'BBQ Grill Set', 'Kitchen', 199.99, 25);
    INSERT INTO products VALUES (8, 'Patio Heater', 'Heating', 159.99, 15);

    -- Insert sample orders
    INSERT INTO orders VALUES (1, 1, '2024-06-01', 389.97, 'delivered');
    INSERT INTO orders VALUES (2, 2, '2024-06-05', 129.99, 'delivered');
    INSERT INTO orders VALUES (3, 3, '2024-06-10', 324.97, 'shipped');
    INSERT INTO orders VALUES (4, 1, '2024-07-01', 199.99, 'delivered');
    INSERT INTO orders VALUES (5, 4, '2024-07-15', 89.99, 'delivered');
    INSERT INTO orders VALUES (6, 5, '2024-07-20', 459.97, 'processing');
    INSERT INTO orders VALUES (7, 2, '2024-08-01', 69.98, 'delivered');
    INSERT INTO orders VALUES (8, 3, '2024-08-10', 159.99, 'shipped');

    -- Insert order items
    INSERT INTO order_items VALUES (1, 1, 1, 1, 89.99);
    INSERT INTO order_items VALUES (2, 1, 2, 1, 299.99);
    INSERT INTO order_items VALUES (3, 2, 5, 1, 129.99);
    INSERT INTO order_items VALUES (4, 3, 3, 5, 24.99);
    INSERT INTO order_items VALUES (5, 3, 4, 2, 45.99);
    INSERT INTO order_items VALUES (6, 4, 7, 1, 199.99);
    INSERT INTO order_items VALUES (7, 5, 1, 1, 89.99);
    INSERT INTO order_items VALUES (8, 6, 2, 1, 299.99);
    INSERT INTO order_items VALUES (9, 6, 8, 1, 159.99);
    INSERT INTO order_items VALUES (10, 7, 6, 2, 34.99);
    INSERT INTO order_items VALUES (11, 8, 8, 1, 159.99);
""")

conn.commit()
conn.close()
print(f"Database created at: {db_path}")
print("Tables: customers, products, orders, order_items")
print("Ready for the agent!")
