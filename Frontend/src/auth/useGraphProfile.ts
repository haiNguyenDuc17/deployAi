import { useEffect, useState } from "react";
import { useMsal } from "@azure/msal-react";
import { InteractionRequiredAuthError, AccountInfo } from "@azure/msal-browser";
import { loginRequest } from "./authConfig";

export type GraphProfile = {
  displayName?: string;
  mail?: string;
  department?: string;
  onPremisesSamAccountName?: string;
  givenName?: string;
  surname?: string;
};

export function useGraphProfile() {
  const { instance, accounts } = useMsal();
  const [profile, setProfile] = useState<GraphProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const account: AccountInfo | undefined = accounts[0];

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      if (!account) return;
      setLoading(true);
      setError(null);
      try {
        // Try silent first
        const tokenResp = await instance.acquireTokenSilent({
          ...loginRequest,
          account,
        });

        const res = await fetch(
          "https://graph.microsoft.com/v1.0/me?$select=displayName,mail,department,onPremisesSamAccountName,givenName,surname",
          { headers: { Authorization: `Bearer ${tokenResp.accessToken}` } }
        );

        if (!res.ok) {
          throw new Error(`Graph error ${res.status}`);
        }

        const json = (await res.json()) as GraphProfile;
        if (!cancelled) setProfile(json);
      } catch (e: any) {
        // If silent fails due to interaction required, fall back to redirect
        if (e instanceof InteractionRequiredAuthError) {
          try {
            await instance.acquireTokenRedirect(loginRequest);
            return; // will redirect; code below won't run now
          } catch (redirectErr: any) {
            if (!cancelled) setError(redirectErr?.message ?? "Redirect failed.");
          }
        } else {
          if (!cancelled) setError(e?.message ?? "Unknown error.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [account, instance]);

  return { profile, loading, error };
}
