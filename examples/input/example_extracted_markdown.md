# XR-2000 Router User Manual

## Introduction

The XR-2000 is a high-performance enterprise router designed for demanding network environments. It provides advanced routing capabilities, robust security features, and exceptional reliability.

### Product Overview

The XR-2000 router offers enterprise-grade performance with up to 10 Gbps throughput and support for up to 500 simultaneous VPN connections.

### Package Contents

Router unit, power adapter, Ethernet cable, mounting brackets, quick start guide, and warranty card.

## Safety Information

Please read all safety instructions before operating the device. Failure to follow these guidelines may result in equipment damage or personal injury.

⚠️ **WARNING**: Disconnect power before servicing. High voltage present. Refer servicing to qualified personnel only.

⚠️ **WARNING**: Do not open the device enclosure. No user-serviceable parts inside. Opening the case will void warranty.

⚠️ **CAUTION**: Ensure adequate ventilation. Do not block air vents. Maintain 2 inches clearance on all sides.

ℹ️ **NOTE**: Configuration changes take effect immediately. Save configuration to flash memory to persist across reboots.

### Electrical Safety

Use only the supplied power adapter. Do not expose the device to moisture or extreme temperatures.

## Technical Specifications

Detailed technical specifications and performance characteristics of the XR-2000 router.

### Hardware Specifications

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Operating Temperature | 0 to 40 | °C | Non-condensing environment |
| Storage Temperature | -20 to 70 | °C | |
| Power Consumption | 45 | W | Maximum under full load |
| Input Voltage | 100-240 | VAC | Auto-sensing |
| Throughput | 10 | Gbps | Full-duplex |
| Routing Table Size | 100,000 | entries | |
| VPN Tunnels | 500 | connections | IPSec and SSL |
| Memory | 4 | GB | DDR4 RAM |

### Port Specifications

**Table 1: Port Specifications**

| Port Type | Quantity | Speed | Description |
|-----------|----------|-------|-------------|
| WAN | 2 | 10 Gbps | Fiber optic uplink ports |
| LAN | 8 | 1 Gbps | Gigabit Ethernet ports |
| Console | 1 | 115200 baud | RJ-45 management port |
| USB | 1 | USB 3.0 | Configuration backup |

### LED Indicators

**Table 2: LED Indicators**

| LED | Color | Status | Meaning |
|-----|-------|--------|---------|
| Power | Green | Solid | Device powered on |
| Power | Off | Off | Device powered off |
| Status | Green | Solid | Normal operation |
| Status | Amber | Blinking | Firmware update in progress |
| WAN | Green | Solid | Link established |
| WAN | Green | Blinking | Data transmission |
| LAN 1-8 | Green | Solid | Port connected |
| LAN 1-8 | Green | Blinking | Port activity |

### Default Configuration

**Table 3: Default Configuration**

| Parameter | Default Value | Notes |
|-----------|---------------|-------|
| IP Address | 192.168.1.1 | Management interface |
| Subnet Mask | 255.255.255.0 | |
| DHCP Server | Enabled | Range: .100-.200 |
| Admin Username | admin | Change on first login |
| Admin Password | admin | Change immediately |

## Installation

Step-by-step instructions for installing and configuring your XR-2000 router.

### Initial Setup Procedure

1. **Unpack the router and verify all components are present**
   - Report missing items to your supplier immediately

2. **Choose a suitable mounting location with adequate ventilation**
   - Ambient temperature should be within 0-40°C

3. **Connect the power adapter to the router and plug into AC outlet**
   - Power LED should illuminate green within 30 seconds

4. **Connect your computer to any LAN port using an Ethernet cable**
   - Use a standard CAT5e or better cable

5. **Open a web browser and navigate to https://192.168.1.1**
   - Accept the security certificate warning on first connection

6. **Login using default credentials (admin/admin)**
   - You will be prompted to change password on first login

7. **Follow the setup wizard to configure WAN connection**
   - Select appropriate connection type (DHCP, Static, PPPoE)

8. **Configure LAN settings and DHCP parameters**
   - Default DHCP range is 192.168.1.100-200

9. **Apply settings and wait for router to reboot**
   - Reboot typically takes 2-3 minutes

10. **Verify internet connectivity and save configuration**
    - Use the 'Save Configuration' button to persist settings

### Factory Reset Procedure

1. **Locate the recessed Reset button on the rear panel**
   - A paperclip or pin is required to press the button

2. **With the router powered on, press and hold the Reset button**
   - Hold for exactly 10 seconds

3. **Release the button when the Status LED begins blinking amber**
   - This indicates reset is in progress

4. **Wait for the router to complete the reset process**
   - All LEDs will cycle and router will reboot (approximately 3 minutes)

5. **Verify reset by logging in with default credentials**
   - All custom settings will be lost

### Firmware Update Procedure

1. **Download the latest firmware from the manufacturer website**
   - Verify firmware file integrity using provided MD5 checksum

2. **Log into the router web interface**
   - Ensure stable power during firmware update

3. **Navigate to System > Firmware Update**

4. **Click 'Choose File' and select the downloaded firmware file**
   - File should have .bin extension

5. **Click 'Upload and Install' to begin the update**
   - Do not power off or reset during this process

6. **Wait for upload and verification to complete**
   - Progress bar will show upload status (typically 2-5 minutes)

7. **Router will automatically reboot after installation**
   - Allow 3-5 minutes for reboot and initialization

8. **Log back in and verify firmware version under System > About**
   - Configuration is preserved during firmware updates

## Diagrams

### Figure 1: Front Panel Layout
![Front Panel](figures/front_panel.png)
*Shows the location of LED indicators, reset button, and port connections on the front panel of the XR-2000 router.*

### Figure 2: Rear Panel Connections
![Rear Panel](figures/rear_panel.png)
*Illustrates the power inlet, WAN ports, ventilation openings, and mounting points on the rear panel.*

### Figure 3: Network Topology Example
![Network Topology](figures/network_topology.png)
*Example network diagram showing typical deployment scenario with XR-2000 connecting multiple VLANs and WAN connections.*

## Troubleshooting

### Common Issues

**No Power LED**
- Check power adapter connection
- Verify AC outlet has power
- Try different power outlet
- Contact support if issue persists

**Cannot Access Web Interface**
- Verify computer is connected to LAN port
- Check computer IP address (should be 192.168.1.x)
- Try accessing via IP: https://192.168.1.1
- Perform factory reset if necessary

**No Internet Connection**
- Verify WAN cable is properly connected
- Check WAN configuration settings
- Verify ISP service is active
- Review firewall and routing settings

## Support

For technical support, please contact:
- Email: support@example.com
- Phone: 1-800-555-0100
- Website: https://support.example.com

---

**Document Information**
- Version: 3.2.1
- Last Updated: September 15, 2024
- Part Number: XR2000-UM-001
