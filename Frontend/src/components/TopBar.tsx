import React from "react";
import { Button } from "primereact/button";
import { useMsal, useIsAuthenticated } from "@azure/msal-react";
import { loginRequest } from "../auth/authConfig"; // adjust path
import { useGraphProfile } from "../auth/useGraphProfile";

interface TopBarProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

const TopBar: React.FC<TopBarProps> = ({ sidebarCollapsed, onToggleSidebar }) => {
  const { instance, accounts } = useMsal();
  const isAuthenticated = useIsAuthenticated();
  const { profile, loading } = useGraphProfile();

  const handleLogin = () => instance.loginRedirect(loginRequest);
  const handleLogout = () => instance.logoutRedirect();

  // Best-effort identity (displayName > mail > username)
  const identity =
    profile?.displayName ||
    profile?.mail ||
    accounts[0]?.username ||
    "User";

  // Tiny avatar initials (optional)
  const initials = (() => {
    const name = profile?.displayName ?? accounts[0]?.username ?? "";
    const parts = name.split(/[.\s_@-]+/).filter(Boolean);
    const first = parts[0]?.[0] ?? "";
    const last = parts[1]?.[0] ?? "";
    return (first + last).toUpperCase() || (name[0]?.toUpperCase() ?? "U");
  })();

  return (
    <div
      className="topbar flex align-items-center justify-content-between px-4 py-3 fixed top-0 right-0 z-4"
      style={{
        backgroundColor: "#1a1a1a",
        borderBottom: "1px solid #404040",
        color: "#ffffff",
        left: sidebarCollapsed ? "80px" : "280px",
        height: "70px",
        transition: "left 0.3s ease",
      }}
    >
      {/* Left Side - Sidebar Toggle */}
      <div className="flex align-items-center gap-4">
        <Button
          icon="pi pi-bars"
          className="p-button-text p-button-rounded"
          style={{ color: "#ffffff" }}
          onClick={onToggleSidebar}
        />
        <span className="text-xl font-bold text-white">BTC Predict Dashboard</span>
      </div>

      {/* Right Side */}
      <div className="flex align-items-center gap-3">
        <Button
          label="Buy / Sell"
          className="p-button-warning p-button-sm"
          style={{ backgroundColor: "#ff8c00", border: "none", borderRadius: "6px" }}
          onClick={() => window.open("https://www.binance.com/", "_blank")}
        />

        {!isAuthenticated ? (
          <Button
            label="Login"
            className="p-button-warning p-button-sm"
            style={{ backgroundColor: "#ff8c00", border: "none", borderRadius: "6px" }}
            onClick={handleLogin}
          />
        ) : (
          <div className="flex align-items-center gap-2">
            {/* Loading shimmer/fallback */}
            {loading ? (
              <span className="text-sm" style={{ opacity: 0.7 }}>
                Loading profile…
              </span>
            ) : (
              <>
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: "50%",
                    background: "#2d2d2d",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                  title={identity}
                >
                  {initials}
                </div>
                <span className="text-sm">{identity}</span>
              </>
            )}

            <Button
              label="Logout"
              className="p-button-warning p-button-sm"
              style={{ backgroundColor: "#ff8c00", border: "none", borderRadius: "6px" }}
              onClick={handleLogout}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default TopBar;
