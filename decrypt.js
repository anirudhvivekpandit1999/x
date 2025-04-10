const crypto = require("crypto");

const secretKey = Buffer.from("qwertyuiopasdfghjklzxcvbnm123456", "utf-8");
const iv = Buffer.alloc(16, 0); // IV filled with zeros

function decryptData(encryptedData) {
    const decipher = crypto.createDecipheriv("aes-256-cbc", secretKey, iv);
    let decrypted = decipher.update(encryptedData, "hex", "utf-8");
    decrypted += decipher.final("utf-8");
    return JSON.parse(decrypted);
}

// Your encrypted string
const encryptedText = "05f63b3a2607ddb0ee5f343d7f612a4e9afe1465267f59ec1a611b7a26a4a8fe7c37b3bb29a8daebb15dda45c31e84e8cad3b2ca206430042a4ef9f6f2ff7798130fd42c13267c59235aebfeae10c6d198349c06eee7df50ce8a89048c8c11f77025cb9ddfd495b4df546d64a797c52d2c599a5644455b55b48ec053d1bec422cd05701cca917b4c8fc5bd2140adeb5c7ad37a24896a71815f794f9abb38f727ee50ec34f999838c2c926d3365b87971c9e8e57251922bc7eece1ae898cd2f12d0350eb561b26b8e001b364539ec3a5f11ecd39240e63d7d1d24a7a0f115457de93232435f19f6d7d265756c9b6efaa63374277c9ab7ed2f7b5ea76152a711401a50a7f9867ed23bb1af2352d1b3be01ddd8d36215de42c90720748d12cf86261f0b67e2c02eea1d377bb2537ccf961f35f24f753eb6f70b905f5426a1c19fb6045cd766d148fe4681282a97b6ec179e011257efff3def67b65368576329983d6415a7db16a78912d99f50f8be0fd524d1fba1fc565f12819e4fa5292f29266d9417a7d82bc166b077e8edddf25e124aa42485372ce106d35e13916ce31ed5aca3160dd64b0bc1be27958271d448dd8d88b8d57811d7d0ccc2b520a8e2867e45a4541d52ba86a1740c01802faf424b7c75822753bae861e44f5050da67d03b0ceccd5b289376754c7f7253878620da35eb7308c609d2d1bd185c3a132e89210e8273b9a352f044814bbefda62506a24aa20ef51f6c0059eeb32981f4125d954fd7ff71d85c315d662e5171a4f39127e42635e7f86bc0eb1ce00b1d911da6932bf66ecccc857a3b88707d09c0f67360afcc5cd7bdaa9c17b431b5172ff917650aa0217675cb184e403bbd3b0beb49d251faa2fb9213618cd014357fa905cf7329ffba4dc09f9a2a669e3cec03c360c97d810af810ec197bc7d75ebe884f30c9ab17b5fdef2c51252b36c3650c6b7df1bff01c07a80a93034e7a5e632fc271c8a4f301e17a94d518ee0c4fa3e96447e3f1285fe80dbf890b8441d90471c01e60ebbda1ffa7b20f5be36605c86ee82ea4f7d0581ee25d649d80e6a6231affe8efa1";

try {
    const decryptedData = decryptData(encryptedText);
    console.log("Decrypted Data:", decryptedData);
} catch (error) {
    console.error("Decryption failed:", error.message);
}
