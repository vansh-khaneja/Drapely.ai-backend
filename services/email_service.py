"""Email service for sending notifications"""
import resend
from typing import Dict
import logging
from config import RESEND_API_KEY, RESEND_FROM_EMAIL, FRONTEND_URL, LOGO_URL

logger = logging.getLogger(__name__)


def get_email_template(user_id: str, email: str, processed_images: Dict[str, str], subscription_type: str, collection: str, frontend_url: str) -> str:
    """Generate HTML email template for try-on completion matching website theme"""
    
    # Customize content based on subscription type
    if subscription_type == "premium":
        plan_description = "Premium subscribers get priority processing and access to exclusive collections!"
        plan_color = "#ff6b9d"
    else:
        plan_description = "Try our premium plan for faster processing and more features!"
        plan_color = "#ff6b9d"
    
    # Generate product images grid (no individual buttons) - using table for email compatibility
    products_html = '<table width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;"><tr>'
    product_count = 0
    for product_id, image_url in processed_images.items():
        if product_count > 0 and product_count % 2 == 0:
            products_html += '</tr><tr>'
        products_html += f"""
        <td width="50%" style="padding: 10px; vertical-align: top;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #f0f0f0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <tr>
                    <td style="padding: 0;">
                        <img src="{image_url}" alt="Try-On Result for {product_id}" style="width: 100%; height: auto; display: block; max-width: 100%;" />
                    </td>
                </tr>
                <tr>
                    <td style="padding: 12px; text-align: center;">
                        <p style="margin: 0; color: #333; font-size: 14px; font-weight: 600;">{product_id}</p>
                    </td>
                </tr>
            </table>
        </td>
        """
        product_count += 1
    # Fill remaining cells if odd number of products
    if product_count % 2 == 1:
        products_html += '<td width="50%" style="padding: 10px;"></td>'
    products_html += '</tr></table>'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Virtual Try-On Complete - DRAPELY.ai</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #fafafa;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fafafa; padding: 40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                        <!-- Header with Logo -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #fff5f7 0%, #ffeef2 100%); padding: 40px 30px; text-align: center; border-bottom: 2px solid #ff6b9d;">
                                <div style="margin-bottom: 20px;">
                                    <img src="{LOGO_URL}" alt="DRAPELY.ai Logo" style="max-width: 180px; height: auto; display: block; margin: 0 auto;" />
                                </div>
                                <div style="display: inline-block; background: #ff6b9d; color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 10px;">
                                    // FUTURISTIC
                                </div>
                                <h1 style="color: #1a1a2e; margin: 10px 0; font-size: 32px; font-weight: 700; letter-spacing: -0.5px;">
                                    Where AI Meets Style
                                </h1>
                            </td>
                        </tr>
                        
                        <!-- Main Content -->
                        <tr>
                            <td style="padding: 40px 30px;">
                                <p style="font-size: 18px; color: #1a1a2e; margin: 0 0 20px 0; font-weight: 600;">
                                    ✨ Your Virtual Try-On is Complete!
                                </p>
                                
                                <p style="font-size: 16px; color: #555; margin: 0 0 25px 0; line-height: 1.8;">
                                    Fashion that learns you. We've successfully processed <strong style="color: #ff6b9d;">{len(processed_images)}</strong> garment(s) from the <strong style="color: #ff6b9d;">{collection}</strong> collection for your personalized try-on experience.
                                </p>
                                
                                <!-- Plan Badge -->
                                <div style="background: linear-gradient(135deg, #ffeef2 0%, #fff5f7 100%); padding: 12px 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {plan_color};">
                                    <p style="margin: 0; color: #1a1a2e; font-size: 14px; font-weight: 600;">
                                        Plan: <span style="color: {plan_color}; text-transform: capitalize;">{subscription_type}</span> • Collection: <span style="color: {plan_color};">{collection}</span> • Total Processed: <span style="color: {plan_color};">{len(processed_images)}</span>
                                    </p>
                                </div>
                                
                                <p style="font-size: 14px; color: #666; margin: 0 0 25px 0; line-height: 1.6;">
                                    {plan_description}
                                </p>
                                
                                <!-- Try-On Results Section -->
                                <div style="margin: 30px 0;">
                                    <h2 style="color: #1a1a2e; font-size: 22px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #ffeef2;">
                                        Your Try-On Results from {collection}
                                    </h2>
                                    {products_html}
                                </div>
                                
                                <!-- CTA Button -->
                                <div style="text-align: center; margin: 40px 0 30px 0;">
                                    <a href="{frontend_url}/products?category={collection}" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #ff6b9d 0%, #ff4757 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3); transition: transform 0.2s;">
                                        View Collection
                                    </a>
                                </div>
                                
                                <!-- Footer -->
                                <div style="margin-top: 40px; padding-top: 30px; border-top: 1px solid #f0f0f0; text-align: center;">
                                    <p style="color: #888; font-size: 14px; margin: 0 0 10px 0;">
                                        Thank you for using <strong style="color: #ff6b9d;">DRAPELY.ai</strong> Virtual Try-On service!
                                    </p>
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">
                                        If you have any questions, please don't hesitate to contact us.
                                    </p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html


def send_completion_email_sync(email: str, user_id: str, processed_images: Dict[str, str], subscription_type: str, collection: str):
    """Send email notification when try-on processing is complete (synchronous for background task)"""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not configured. Skipping email notification.")
        return
    
    try:
        # Set API key (per Resend docs)
        resend.api_key = RESEND_API_KEY
        
        subject = f"Your Virtual Try-On Results Are Ready! ({subscription_type.title()} Plan - {collection})"
        html_content = get_email_template(user_id, email, processed_images, subscription_type, collection, FRONTEND_URL)
        
        # Send email via Resend (per FastAPI docs)
        params: resend.Emails.SendParams = {
            "from": RESEND_FROM_EMAIL,
            "to": [email],
            "subject": subject,
            "html": html_content,
        }
        
        email_response = resend.Emails.send(params)
        logger.info(f"Email sent successfully to {email}. Email ID: {email_response.get('id')}")
        
    except Exception as e:
        logger.error(f"Failed to send email to {email}: {str(e)}")
        # Don't raise exception - email failure shouldn't break the API response


def send_error_email_sync(email: str, user_id: str, error_message: str, subscription_type: str, collection: str, frontend_url: str):
    """Send error email notification when try-on processing fails"""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not configured. Skipping error email notification.")
        return
    
    try:
        # Set API key (per Resend docs)
        resend.api_key = RESEND_API_KEY
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Virtual Try-On Processing Error</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #fafafa;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fafafa; padding: 40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                        <!-- Header with Logo -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #fff5f7 0%, #ffeef2 100%); padding: 40px 30px; text-align: center; border-bottom: 2px solid #ff6b9d;">
                                <div style="margin-bottom: 20px;">
                                    <img src="{LOGO_URL}" alt="DRAPELY.ai Logo" style="max-width: 180px; height: auto; display: block; margin: 0 auto;" />
                                </div>
                                <h1 style="color: #1a1a2e; margin: 10px 0; font-size: 28px; font-weight: 700;">⚠️ Processing Error</h1>
                            </td>
                        </tr>
                        
                        <!-- Main Content -->
                        <tr>
                            <td style="padding: 40px 30px;">
                
                <p style="font-size: 16px; color: #555;">Hello,</p>
                
                <p style="font-size: 16px; color: #555;">
                    We encountered an error while processing your virtual try-on request for the <strong>{collection}</strong> collection. 
                    Please try again or contact support if the issue persists.
                </p>
                
                <div style="background: #fff5f7; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ff6b9d;">
                    <p style="margin: 0; color: #1a1a2e;">
                        <strong>Plan:</strong> <span style="color: #ff6b9d; text-transform: capitalize;">{subscription_type}</span><br>
                        <strong>Collection:</strong> <span style="color: #ff6b9d;">{collection}</span><br>
                        <strong>Error:</strong> {error_message}
                    </p>
                </div>
                
                                <!-- CTA Button -->
                                <div style="text-align: center; margin: 30px 0;">
                                    <a href="{frontend_url}/products?category={collection}" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #ff6b9d 0%, #ff4757 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);">
                                        Try Again
                                    </a>
                                </div>
                                
                                <!-- Footer -->
                                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #f0f0f0; text-align: center;">
                                    <p style="color: #888; font-size: 14px; margin: 0 0 10px 0;">
                                        Please try submitting your request again.
                                    </p>
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">
                                        If the problem continues, please contact our support team.
                                    </p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        </body>
        </html>
        """
        
        params: resend.Emails.SendParams = {
            "from": RESEND_FROM_EMAIL,
            "to": [email],
            "subject": f"Virtual Try-On Processing Error ({subscription_type.title()} Plan - {collection})",
            "html": html_content,
        }
        
        email_response = resend.Emails.send(params)
        logger.info(f"Error email sent successfully to {email}. Email ID: {email_response.get('id')}")
        
    except Exception as e:
        logger.error(f"Failed to send error email to {email}: {str(e)}")

