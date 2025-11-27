from typing import List, Dict

def optimize_flight_delays(flights: List[Dict]) -> List[Dict]:
    """
    Input: flights = [
        {
            "flight": "DL101",
            "predicted_delay": 22.3,
            "risk": True,
            "anomaly": {"weather": True, "crew_timeout": False, "fuel": True}
        }
    ]
    Output: each flight dict will include delay_flag, early_flag, high_risk_flag, and recommendations
    """
    results = []

    for flight in flights:
        predicted_delay = flight.get("predicted_delay", 0)
        risk = flight.get("risk", False)
        anomaly = flight.get("anomaly", {})

        # Flags
        delay_flag = predicted_delay > 0
        early_flag = predicted_delay < 0
        high_risk_flag = risk

        # Recommendations
        recommendations = []

        # ---------------------------
        # Delayed flight suggestions
        # ---------------------------
        if delay_flag:
            recommendations.extend([
                "Potential delay detected. Monitor flight status closely.",
                "Consider reallocating ground staff to boarding gates.",
                "Prepare alternative gates to reduce congestion.",
                "Check staffing levels during peak hours to avoid bottlenecks.",
                "Notify passengers of estimated delay via app or email.",
                "Coordinate with baggage handling to avoid pile-up.",
                "Check if connecting flights may be affected and update schedules."
            ])

        # ---------------------------
        # Early flight suggestions
        # ---------------------------
        if early_flag:
            recommendations.extend([
                "Flight may arrive early. Notify ground crew to prepare.",
                "Prepare baggage claim and gate early.",
                "Coordinate with gate staff to speed up turnaround.",
                "Check if connecting flights may be affected.",
                "Update passenger notifications for early arrival.",
                "Adjust ground staff assignments to match early schedule."
            ])

        # ---------------------------
        # High risk flight suggestions
        # ---------------------------
        if high_risk_flag:
            recommendations.extend([
                "High risk detected. Enable real-time notifications for staff and passengers.",
                "Reserve emergency resources and check contingency plans.",
                "Check weather conditions and possible operational disruptions.",
                "Verify fuel availability and crew readiness.",
                "Prepare contingency gates and standby staff.",
                "Ensure maintenance teams are on alert for unexpected issues.",
                "Monitor nearby airport traffic to avoid congestion."
            ])

        # ---------------------------
        # Anomaly-based recommendations
        # ---------------------------
        if anomaly.get("weather"):
            recommendations.extend([
                "Weather anomaly detected. Plan contingencies and monitor conditions.",
                "Check weather conditions and possible operational disruptions.",
                "Adjust gate assignments to avoid congestion due to delays.",
                "Notify ground crew and passengers about potential delays.",
                "Coordinate with air traffic control for alternate routing.",
                "Prepare standby equipment for de-icing or other weather needs.",
                "Monitor nearby airport traffic to avoid bottlenecks."
            ])
        if anomaly.get("crew_timeout"):
            recommendations.extend([
                "Crew timeout detected. Verify crew availability and adjust shifts.",
                "Check legal rest requirements for crew members.",
                "Reassign flights if necessary to avoid crew shortage.",
                "Coordinate with crew scheduling team for backup personnel.",
                "Ensure maintenance teams are on alert for unexpected issues.",
                "Notify affected passengers and staff about potential delays.",
                "Monitor crew workload to prevent future overtime issues."
            ])
        if anomaly.get("fuel"):
            recommendations.extend([
                "Fuel issue detected. Ensure refueling schedule is on track and backup fuel is ready.",
                "Verify fuel availability for all upcoming flights.",
                "Coordinate with ground staff to prioritize refueling critical flights.",
                "Check if alternative fuel suppliers are available in case of shortage.",
                "Prepare contingency gates and standby staff in case of delay.",
                "Notify operations control and pilots about fuel constraints.",
                "Monitor fuel consumption for incoming flights to prevent cascading delays."
            ])

        # ---------------------------
        # On-time flight
        # ---------------------------
        if not delay_flag and not early_flag and not high_risk_flag and not any(anomaly.values()):
            recommendations.append("Flight is on time, no action needed.")

        # Build result dictionary
        result = {
            "flight": flight.get("flight"),
            "predicted_delay": predicted_delay,
            "delay_flag": delay_flag,
            "early_flag": early_flag,
            "high_risk_flag": high_risk_flag,
            "anomaly": anomaly,
            "recommendations": recommendations
        }

        results.append(result)

    return results

