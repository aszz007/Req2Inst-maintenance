#!/usr/bin/env python3
"""Generate synthetic PlantUML use-case diagrams and metadata."""

import subprocess
import shutil
import urllib.request
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import random


class PlantUMLGenerator:
    """Generate synthetic PlantUML use-case diagrams."""

    def __init__(self, output_dir: Path):
        """Initialize the instance."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plantuml_jar = Path(__file__).parent.parent / "plantuml.jar"

        self.domains = self._init_domain_pools()

        self.complexity_config = {
            'simple': {'min_usecases': 5, 'max_usecases': 8, 'min_actors': 1, 'max_actors': 2, 'count': 75},
            'medium': {'min_usecases': 10, 'max_usecases': 15, 'min_actors': 2, 'max_actors': 4, 'count': 60},
            'complex': {'min_usecases': 18, 'max_usecases': 25, 'min_actors': 3, 'max_actors': 6, 'count': 15}
        }

    def _init_domain_pools(self) -> dict:
        """Initialize domain-specific generation pools."""
        return {
            "ecommerce": self._get_ecommerce_pool(),
            "authentication": self._get_authentication_pool(),
            "content_management": self._get_content_management_pool(),
            "social_interaction": self._get_social_interaction_pool(),
            "customer_service": self._get_customer_service_pool(),
            "data_analysis": self._get_data_analysis_pool(),
            "permission_management": self._get_permission_management_pool(),
            "notification_system": self._get_notification_system_pool(),
            "file_management": self._get_file_management_pool(),
            "booking_system": self._get_booking_system_pool()
        }

    def _get_ecommerce_pool(self) -> dict:
        """Return ecommerce pool."""
        return {
            'actors': [
                "Customer", "Guest", "Admin", "Seller", "Buyer",
                "Payment Gateway", "Shipping Service", "Inventory System",
                "Marketing Manager", "Support Agent", "Warehouse Staff"
            ],
            'usecases': [
                "Browse Products", "Search Items", "View Details", "Filter Results",
                "Add to Cart", "Remove from Cart", "Update Quantity", "Save for Later",
                "Checkout", "Make Payment", "Apply Coupon", "Calculate Tax",
                "Confirm Order", "Track Order", "Cancel Order", "Return Item",
                "Write Review", "Rate Product", "Upload Photos", "Report Issue",
                "Add to Wishlist", "Share Wishlist", "Compare Products",
                "Manage Inventory", "Update Stock", "Set Pricing", "Create Promotion",
                "Process Refund", "Handle Dispute", "Generate Invoice", "Send Notification",
                "Manage Categories", "Update Product Info", "Bulk Import", "Export Data"
            ],
            'scenarios': [
                "Product Discovery", "Shopping Cart", "Checkout Flow", "Order Management",
                "Customer Service", "Inventory Control", "Marketing Campaign", "Payment Processing"
            ]
        }

    def _get_authentication_pool(self) -> dict:
        """Return authentication pool."""
        return {
            'actors': [
                "User", "Admin", "Guest", "System Administrator",
                "OAuth Provider", "Email Service", "SMS Gateway",
                "Security Manager", "Audit System", "Token Service"
            ],
            'usecases': [
                "Login", "Logout", "Register", "Validate Credentials",
                "Generate Token", "Refresh Token", "Revoke Token",
                "Verify Email", "Send Verification", "Confirm Account",
                "Request Reset", "Verify Identity", "Update Password",
                "Enable 2FA", "Disable 2FA", "Send OTP", "Verify OTP",
                "Login with OAuth", "Authorize App", "Link Account", "Unlink Account",
                "Manage Sessions", "Track Activity", "Detect Anomaly",
                "Lock Account", "Unlock Account", "Force Logout",
                "Update Profile", "Change Email", "Verify Phone",
                "Set Security Questions", "Remember Device", "Clear Sessions"
            ],
            'scenarios': [
                "User Login", "Registration", "Password Management", "Two Factor Auth",
                "Social Login", "Session Management", "Security Monitoring", "Account Recovery"
            ]
        }

    def _get_content_management_pool(self) -> dict:
        """Return content management pool."""
        return {
            'actors': [
                "Author", "Editor", "Moderator", "Admin", "Reviewer",
                "Publisher", "Content Creator", "SEO Specialist",
                "Media Manager", "Workflow Manager", "Reader"
            ],
            'usecases': [
                "Create Article", "Edit Document", "Delete Content", "Restore Version",
                "Submit for Review", "Approve", "Reject", "Publish", "Unpublish",
                "Upload Media", "Compress Image", "Generate Thumbnail", "Optimize File",
                "Review Content", "Flag Inappropriate", "Remove Content", "Archive",
                "Save Version", "Compare Versions", "Rollback Changes",
                "Assign Task", "Set Deadline", "Monitor Progress", "Send Reminder",
                "Tag Content", "Categorize", "Set Metadata", "Add Keywords",
                "Schedule Publication", "Auto-Publish", "Manage Draft",
                "Collaborate", "Add Comment", "Track Changes", "Resolve Conflict",
                "Export Content", "Import Content", "Bulk Operations"
            ],
            'scenarios': [
                "Article Publishing", "Media Management", "Content Moderation",
                "Version Control", "Workflow Management", "Collaboration", "SEO Optimization"
            ]
        }

    def _get_social_interaction_pool(self) -> dict:
        """Return social interaction pool."""
        return {
            'actors': [
                "User", "Friend", "Follower", "Group Admin", "Moderator",
                "Content Creator", "Advertiser", "Influencer",
                "Notification Service", "Recommendation Engine", "Analytics System"
            ],
            'usecases': [
                "Follow User", "Unfollow User", "Send Request", "Accept Request", "Decline Request",
                "Create Post", "Edit Post", "Delete Post", "Share Post", "Repost",
                "Like Post", "Unlike", "React", "Bookmark",
                "Comment", "Reply", "Delete Comment", "Report Comment",
                "Send Message", "Receive Message", "Encrypt Message", "Delete Chat",
                "Create Group", "Join Group", "Leave Group", "Invite Member",
                "Post in Group", "Moderate Group", "Set Rules",
                "Go Live", "Stream Video", "Watch Stream", "Send Gift",
                "Create Story", "View Story", "React to Story",
                "Tag User", "Mention", "Check-in Location",
                "Block User", "Unblock User", "Report User", "Mute Notifications",
                "Customize Privacy", "Hide Post", "Filter Feed"
            ],
            'scenarios': [
                "User Connection", "Post Interaction", "Direct Messaging", "Group Activity",
                "Live Streaming", "Story Feature", "Privacy Management", "Content Moderation"
            ]
        }

    def _get_customer_service_pool(self) -> dict:
        """Return customer service pool."""
        return {
            'actors': [
                "Customer", "Support Agent", "Supervisor", "Manager",
                "Chatbot", "Ticketing System", "Knowledge Base",
                "QA Team", "Escalation Handler", "Technical Support"
            ],
            'usecases': [
                "Create Ticket", "Assign Ticket", "Update Status", "Close Ticket",
                "Escalate Issue", "Transfer Ticket", "Merge Tickets",
                "Chat with Agent", "Send Message", "Upload Attachment",
                "Search Articles", "View Solution", "Rate Helpfulness",
                "Submit Feedback", "Rate Service", "Provide Suggestions",
                "Make Call", "Route Call", "Record Call", "End Call",
                "Track History", "View Conversation", "Export Chat",
                "Set Priority", "Add Tags", "Assign Category",
                "Request Callback", "Schedule Appointment", "Send Follow-up",
                "Generate Report", "Analyze Metrics", "Monitor Performance"
            ],
            'scenarios': [
                "Ticket Management", "Live Chat Support", "Knowledge Base",
                "Feedback Collection", "Call Center", "Issue Resolution", "Performance Monitoring"
            ]
        }

    def _get_data_analysis_pool(self) -> dict:
        """Return data analysis pool."""
        return {
            'actors': [
                "Analyst", "Data Scientist", "Manager", "Admin",
                "User", "Report Viewer", "Dashboard Creator",
                "BI Specialist", "Data Engineer", "Stakeholder"
            ],
            'usecases': [
                "Select Data", "Apply Filters", "Generate Report", "Export Report",
                "Export Data", "Choose Format", "Schedule Export", "Batch Export",
                "View Dashboard", "Customize Widgets", "Refresh Data", "Share Dashboard",
                "Create Chart", "Select Metrics", "Share Visualization", "Embed Chart",
                "Run Query", "Optimize Query", "Save Query", "Schedule Query",
                "Analyze Trends", "Detect Anomalies", "Predict Outcomes",
                "Create Dataset", "Join Tables", "Transform Data", "Clean Data",
                "Set Alerts", "Configure Threshold", "Receive Notification",
                "Manage Access", "Grant Permissions", "Audit Usage",
                "Import Data", "Validate Data", "Archive Data"
            ],
            'scenarios': [
                "Report Generation", "Data Export", "Dashboard View", "Data Visualization",
                "Query Execution", "Trend Analysis", "Data Management", "Access Control"
            ]
        }

    def _get_permission_management_pool(self) -> dict:
        """Return permission management pool."""
        return {
            'actors': [
                "Admin", "User", "Manager", "Security Officer",
                "System Administrator", "Auditor", "Role Manager",
                "Resource Owner", "Access Controller", "Compliance Officer"
            ],
            'usecases': [
                "Create Role", "Edit Role", "Delete Role", "Assign Role",
                "Grant Permission", "Revoke Permission", "Check Access",
                "Create User", "Deactivate User", "Update User Info",
                "Assign Resource", "Remove Resource", "Share Resource",
                "Set Policy", "Update Policy", "Enforce Policy",
                "View Audit Log", "Export Log", "Generate Report",
                "Request Access", "Approve Request", "Deny Request",
                "Delegate Authority", "Inherit Permissions", "Override Rule",
                "Create Group", "Add Member", "Remove Member",
                "Set Expiry", "Renew Access", "Auto-Revoke",
                "Encrypt Data", "Decrypt Data", "Mask Sensitive Info",
                "Monitor Activity", "Detect Violation", "Alert Admin"
            ],
            'scenarios': [
                "Role Management", "Permission Control", "User Management",
                "Resource Access", "Policy Enforcement", "Audit Trail",
                "Access Request", "Group Management", "Security Monitoring"
            ]
        }

    def _get_notification_system_pool(self) -> dict:
        """Return notification system pool."""
        return {
            'actors': [
                "User", "System", "Admin", "Service",
                "Email Gateway", "SMS Gateway", "Push Server",
                "Scheduler", "Template Engine", "Analytics Service"
            ],
            'usecases': [
                "Send Email", "Send SMS", "Send Push", "Send In-App",
                "Configure Preferences", "Enable Channel", "Disable Channel",
                "Create Template", "Edit Template", "Activate Template",
                "Schedule Notification", "Cancel Schedule", "Reschedule",
                "Track Delivery", "Check Status", "Retry Failed",
                "Subscribe Topic", "Unsubscribe", "Manage Subscriptions",
                "Set Frequency", "Configure Quiet Hours", "Apply Rules",
                "Batch Send", "Personalize Message", "Add Attachment",
                "Generate Preview", "Test Notification", "A/B Test",
                "View History", "Export Logs", "Analyze Performance",
                "Filter Spam", "Block Sender", "Whitelist",
                "Trigger Alert", "Escalate Priority", "Send Reminder"
            ],
            'scenarios': [
                "Multi-Channel Delivery", "Preference Management", "Template Management",
                "Scheduling", "Delivery Tracking", "Subscription Management",
                "Batch Operations", "Analytics", "Spam Control"
            ]
        }

    def _get_file_management_pool(self) -> dict:
        """Return file management pool."""
        return {
            'actors': [
                "User", "Admin", "Collaborator", "Viewer",
                "Storage Service", "Sync Service", "Search Engine",
                "Backup Service", "Virus Scanner", "Media Processor"
            ],
            'usecases': [
                "Upload File", "Download File", "Delete File", "Restore File",
                "Create Folder", "Rename Folder", "Move Folder", "Delete Folder",
                "Share File", "Revoke Access", "Set Permissions", "Get Link",
                "Search Files", "Filter Results", "Sort Files", "Tag File",
                "Preview File", "View Metadata", "Edit Online", "Comment",
                "Version Control", "Restore Version", "Compare Versions",
                "Sync Files", "Auto-Upload", "Selective Sync",
                "Compress File", "Extract Archive", "Convert Format",
                "Scan Virus", "Quarantine File", "Clean File",
                "Backup Data", "Schedule Backup", "Restore Backup",
                "Generate Thumbnail", "Process Media", "Extract Text",
                "Set Expiry", "Archive Old Files", "Free Space",
                "Track Changes", "View Activity", "Export Report"
            ],
            'scenarios': [
                "File Operations", "Folder Management", "Sharing", "Search",
                "Preview and Edit", "Version Control", "Synchronization",
                "Security Scan", "Backup and Restore", "Storage Management"
            ]
        }

    def _get_booking_system_pool(self) -> dict:
        """Return booking system pool."""
        return {
            'actors': [
                "Customer", "Admin", "Service Provider", "Guest",
                "Payment Gateway", "Calendar Service", "Notification Service",
                "Resource Manager", "Scheduler", "Analytics Service"
            ],
            'usecases': [
                "Search Availability", "View Calendar", "Select Time", "Check Slot",
                "Make Booking", "Confirm Booking", "Cancel Booking", "Reschedule",
                "Process Payment", "Refund Payment", "Apply Discount",
                "Receive Confirmation", "Send Reminder", "Update Status",
                "Check-in", "Check-out", "No-show", "Late Arrival",
                "Manage Slots", "Set Capacity", "Block Dates", "Open Slots",
                "Create Service", "Update Service", "Set Pricing", "Add Rules",
                "View Bookings", "Filter List", "Export Data", "Generate Report",
                "Rate Service", "Leave Review", "Respond to Review",
                "Send Notification", "Automated Reminder", "Follow-up Email",
                "Waitlist Add", "Waitlist Notify", "Auto-Book",
                "Group Booking", "Multi-Day Reservation", "Recurring Booking",
                "Track Occupancy", "Forecast Demand", "Optimize Schedule"
            ],
            'scenarios': [
                "Booking Process", "Payment Handling", "Schedule Management",
                "Resource Management", "Customer Communication", "Check-in/out",
                "Reviews and Ratings", "Waitlist Management", "Analytics"
            ]
        }

    def _generate_random_diagram(self, domain_name: str, pool: dict,
                                 complexity: str, index: int) -> Tuple[str, str, List[str], List[str], List[Tuple]]:
        """Generate random diagram."""
        config = self.complexity_config[complexity]

        num_actors = random.randint(config['min_actors'], config['max_actors'])
        num_usecases = random.randint(config['min_usecases'], config['max_usecases'])

        selected_actors = random.sample(pool['actors'], min(num_actors, len(pool['actors'])))
        selected_usecases = random.sample(pool['usecases'], min(num_usecases, len(pool['usecases'])))

        scenario_name = random.choice(pool['scenarios'])
        file_id = f"{domain_name}_{complexity}_{index}"

        relationships = self._generate_relationships(selected_actors, selected_usecases)

        return file_id, scenario_name, selected_actors, selected_usecases, relationships

    def _generate_relationships(self, actors: List[str], usecases: List[str]) -> List[Tuple]:
        """Generate relationships."""
        relationships = []

        used_usecases = set()

        for actor in actors:
            num_associations = random.randint(1, min(3, len(usecases)))
            for _ in range(num_associations):
                usecase = random.choice(usecases)
                relationships.append((actor, usecase, "association"))
                used_usecases.add(usecase)

        for usecase in usecases:
            if usecase not in used_usecases and random.random() < 0.5:
                actor = random.choice(actors)
                relationships.append((actor, usecase, "association"))
                used_usecases.add(usecase)

        num_includes = random.randint(1, max(1, len(usecases) // 4))
        for _ in range(num_includes):
            if len(usecases) >= 2:
                from_uc, to_uc = random.sample(usecases, 2)
                relationships.append((from_uc, to_uc, random.choice(["include", "extend"])))

        return relationships

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a generated name."""
        name = name.replace('"', '')
        name = name.replace("'", '')
        name = name.replace('\\', '')
        return name

    def generate_plantuml_code(self, scenario_name: str, actors: List[str],
                               use_cases: List[str], relationships: List[Tuple]) -> str:
        """Generate plantuml code."""
        scenario_name = self._sanitize_name(scenario_name)
        actors = [self._sanitize_name(a) for a in actors]
        use_cases = [self._sanitize_name(uc) for uc in use_cases]

        code_lines = ["@startuml", "left to right direction"]

        for actor in actors:
            code_lines.append(f"actor \"{actor}\" as {actor.replace(' ', '_')}")

        code_lines.append("")
        code_lines.append(f"rectangle \"{scenario_name}\" {{")

        for uc in use_cases:
            uc_id = uc.replace(' ', '_').replace('-', '_').replace('/', '_')
            code_lines.append(f"  usecase \"{uc}\" as UC_{uc_id}")

        code_lines.append("}")
        code_lines.append("")

        for from_elem, to_elem, rel_type in relationships:
            from_elem = self._sanitize_name(from_elem)
            to_elem = self._sanitize_name(to_elem)

            from_id = from_elem.replace(' ', '_').replace('-', '_').replace('/', '_')
            to_id = to_elem.replace(' ', '_').replace('-', '_').replace('/', '_')

            if rel_type == "association":
                code_lines.append(f"{from_id} --> UC_{to_id}")
            elif rel_type == "include":
                code_lines.append(f"UC_{from_id} ..> UC_{to_id} : <<include>>")
            elif rel_type == "extend":
                code_lines.append(f"UC_{from_id} ..> UC_{to_id} : <<extend>>")
            elif rel_type == "generalization":
                code_lines.append(f"{from_id} --|> {to_id}")

        code_lines.append("@enduml")

        return "\n".join(code_lines)

    def check_java(self) -> bool:
        """Check java."""
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def download_plantuml(self) -> bool:
        """Download the PlantUML runtime."""
        if self.plantuml_jar.exists():
            print(f"PlantUML already exists: {self.plantuml_jar}")
            return True

        print("Downloading PlantUML...")
        url = "https://github.com/plantuml/plantuml/releases/download/v1.2024.3/plantuml-1.2024.3.jar"

        try:
            urllib.request.urlretrieve(url, self.plantuml_jar)
            print(f"PlantUML downloaded to: {self.plantuml_jar}")
            return True
        except Exception as e:
            print(f"Failed to download PlantUML: {e}")
            return False

    def generate_png(self, puml_file: Path) -> Tuple[bool, str]:
        """Generate png."""
        try:
            result = subprocess.run(
                [
                    "java", "-jar", str(self.plantuml_jar),
                    "-tpng",
                    "-Sdpi=150",
                    str(puml_file)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True, ""
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return False, error_msg
        except subprocess.TimeoutExpired:
            return False, "Timeout (>30s)"
        except FileNotFoundError as e:
            return False, f"File not found: {e}"
        except Exception as e:
            return False, str(e)

    def generate_all(self, target_count: int = 1500):
        """Generate all."""
        if not self.check_java():
            print("Error: Java is not installed. Please install Java 8+ first.")
            sys.exit(1)

        if not self.download_plantuml():
            print("Error: Failed to download PlantUML.")
            sys.exit(1)

        print(f"\nGenerating {target_count} use case diagrams...")
        print(f"Output directory: {self.output_dir}")
        print(f"Strategy: 10 domains × 150 diagrams each (Simple: 75, Medium: 60, Complex: 15)")
        print("=" * 80)

        generated_count = 0
        failed_count = 0
        diagrams_per_domain = target_count // 10
        first_error_shown = False

        for domain_name, pool in self.domains.items():
            domain_dir = self.output_dir / domain_name
            domain_dir.mkdir(exist_ok=True)

            print(f"\n[{domain_name.upper()}] Generating {diagrams_per_domain} diagrams...")

            domain_generated = 0

            for complexity, config in self.complexity_config.items():
                print(f"  Generating {config['count']} {complexity} diagrams...")

                for i in range(config['count']):
                    file_id, scenario_name, actors, usecases, relationships = \
                        self._generate_random_diagram(domain_name, pool, complexity, i)

                    puml_code = self.generate_plantuml_code(scenario_name, actors, usecases, relationships)

                    puml_file = domain_dir / f"{file_id}.puml"
                    with open(puml_file, "w", encoding="utf-8") as f:
                        f.write(puml_code)

                    success, error_msg = self.generate_png(puml_file)

                    if success:
                        puml_file.unlink()

                        domain_generated += 1
                        generated_count += 1

                        if domain_generated % 10 == 0:
                            print(f"    Progress: {domain_generated}/{diagrams_per_domain} diagrams")
                    else:
                        failed_count += 1
                        if not first_error_shown:
                            print(f"\n    ERROR: Failed to generate {puml_file.name}")
                            print(f"    Error message: {error_msg}")
                            print(f"    PUML file preserved at: {puml_file}")
                            print(f"    Please check the file for syntax errors.")
                            print(f"    (Further errors will be counted but not displayed)\n")
                            first_error_shown = True

        print("\n" + "=" * 80)
        print(f"Generation complete!")
        print(f"Successfully generated: {generated_count} diagrams")
        print(f"Failed: {failed_count} diagrams")
        if failed_count > 0:
            print(f"\nFailed PUML files have been preserved for debugging.")
            print(f"Check the output directory for files with .puml extension.")
        print(f"Output location: {self.output_dir}")
        print(f"File format: PNG only (PUML source files deleted for successful generations)")
        print("=" * 80)


def main():
    """Run the command-line entry point."""
    try:
        from config.settings import get_path_config

        path_config = get_path_config()
        output_dir = path_config.PLANT_UML_DIR

    except ImportError:
        print("Warning: Cannot import config.settings, using default path")
        output_dir = Path(__file__).parent.parent / "data" / "raw" / "uml" / "plantuml_usecase"

    generator = PlantUMLGenerator(output_dir)

    target_count = 1500
    generator.generate_all(target_count=target_count)


if __name__ == "__main__":
    main()
