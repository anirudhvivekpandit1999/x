const { Sequelize } = require("sequelize");

// Replace with your actual database credentials
const sequelize = new Sequelize("DB_Metcoke", "sqladmin", "virtualmachine@123", {
  host: "192.168.10.31",
  dialect: "mssql",
  dialectOptions: {
    options: { encrypt: false } // Set to true if using Azure SQL
  },
  logging: false
});

// Test connection
sequelize.authenticate()
  .then(() => console.log("✅ Connected to MSSQL"))
  .catch(err => console.error("❌ Database connection error:", err));

module.exports = sequelize;
