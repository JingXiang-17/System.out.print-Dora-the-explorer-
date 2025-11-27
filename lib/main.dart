// main.dart
import 'dart:convert';
import 'dart:math';

import 'package:csv/csv.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const RunwayApp());
}

class RunwayApp extends StatelessWidget {
  const RunwayApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Runway AI',
      debugShowCheckedModeBanner: false,
      home: const Dashboard(),
      routes: {'/optimize': (context) => const OptimizationPage()},
    );
  }
}

/* =========================
   Helpers (case-insensitive lookup)
   ========================= */

String getVal(Map<String, String> row, String key) {
  // look up case-insensitively among the map's keys
  final target = key.toUpperCase();
  for (final entry in row.entries) {
    if (entry.key.toUpperCase() == target) {
      return entry.value.trim();
    }
  }
  return '';
}

String getOrigin(Map<String, String> r) {
  final candidates = [
    'ORIGIN',
    'ORIG',
    'SOURCE',
    'DEPARTURE',
    'DEPART',
    'FROM',
    'ORIGIN_AIRPORT',
    'ORIGIN_CITY',
  ];
  for (final c in candidates) {
    final v = getVal(r, c);
    if (v.isNotEmpty) return v;
  }
  return '';
}

String getDestination(Map<String, String> r) {
  // IMPORTANT: we will prioritize DESTINATION_AIRPORT per your request
  final candidates = [
    'DESTINATION_AIRPORT',
    'DEST',
    'DEST_AIRPORT',
    'DEST_CITY',
    'ARRIVAL',
    'TO',
  ];
  for (final c in candidates) {
    final v = getVal(r, c);
    if (v.isNotEmpty) return v;
  }
  return '';
}

String getFlightTail(Map<String, String> r) {
  final tailKeys = [
    'TAIL_NUMBER',
    'TAIL',
    'TAIL_NO',
    'TAILNUM',
    'TAIL_NUM',
    'N_NUMBER',
    'REG',
    'REGISTRATION',
  ];
  for (final k in tailKeys) {
    final v = getVal(r, k);
    if (v.isNotEmpty) return v;
  }
  // fallback to flight number fields
  final fkeys = [
    'MKT_CARRIER_FL_NUM',
    'FLIGHT_NUMBER',
    'FL_NUM',
    'FLT_NO',
    'FLIGHT',
  ];
  for (final k in fkeys) {
    final v = getVal(r, k);
    if (v.isNotEmpty) return v;
  }
  return '';
}

/* =========================
   Shared Bottom Navigation Widget
   ========================= */
class BottomNav extends StatelessWidget {
  final int currentIndex;
  final List<Map<String, String>>? datasetRows;
  final Map<String, String>? flightRow;

  const BottomNav({
    super.key,
    required this.currentIndex,
    this.datasetRows,
    this.flightRow,
  });

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: (index) {
        if (index == currentIndex) return;
        if (index == 0) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (_) => const Dashboard()),
          );
        } else if (index == 1) {
          // Anomaly requires a selected flight
          if (datasetRows != null && flightRow != null) {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (c) => AnomalyPage(
                  datasetRows: datasetRows!,
                  flightRow: flightRow!,
                ),
              ),
            );
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Select a flight first to view anomalies'),
              ),
            );
          }
        } else if (index == 2) {
          Navigator.pushReplacementNamed(context, '/optimize');
        }
      },
      items: const [
        BottomNavigationBarItem(
          icon: Icon(Icons.dashboard),
          label: 'Dashboard',
        ),
        BottomNavigationBarItem(icon: Icon(Icons.warning), label: 'Anomaly'),
        BottomNavigationBarItem(icon: Icon(Icons.tune), label: 'Optimize'),
      ],
    );
  }
}

/* =========================
   Dashboard
   ========================= */
class Dashboard extends StatefulWidget {
  const Dashboard({super.key});

  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  bool csvUploaded = false;
  List<String> headers = [];
  List<Map<String, String>> rows = [];

  // summary
  int totalFlights = 0;
  int totalAirlines = 0;
  int totalDestinations = 0;

  // dropdown choices
  List<String> flightTails = [];
  List<String> flightRoutes = [];

  String? selectedFlightTail;
  String? selectedFlightRoute;

  Future<void> pickCsvFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv'],
      withData: true,
    );
    if (result == null) return;
    final bytes = result.files.first.bytes;
    if (bytes == null) return;
    final content = utf8.decode(bytes);
    parseCsv(content);
  }

  void parseCsv(String csvContent) {
    // using csv converter
    final converter = CsvToListConverter(eol: '\n');
    final List<List<dynamic>> raw = converter.convert(csvContent);

    if (raw.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('CSV contains no rows')));
      return;
    }

    // headers preserved as-is for display; detection uses getVal()
    headers = raw.first.map((e) => e.toString().trim()).toList();

    // build rows (map header -> value)
    final dataRows = raw.skip(1).toList();
    rows = dataRows.map((r) {
      final map = <String, String>{};
      for (var i = 0; i < headers.length; i++) {
        final key = headers[i];
        final value = (i < r.length) ? (r[i] ?? '').toString().trim() : '';
        map[key] = value;
      }
      return map;
    }).toList();

    computeSummaries();

    setState(() {
      csvUploaded = true;
    });
  }

  void computeSummaries() {
    totalFlights = rows.length;

    // Airlines: unique set (try a few common columns)
    final airlines = <String>{};
    final airlineCandidates = [
      'DESTINATION_AIRPORT',
    ]; // placeholder, but not airline - we'll look for carrier fields first
    // more appropriate airline keys:
    final carrierKeys = [
      'MKT_UNIQUE_CARRIER',
      'CARRIER',
      'AIRLINE',
      'OP_UNIQUE_CARRIER',
      'OP_CARRIER',
    ];
    for (final r in rows) {
      for (final k in carrierKeys) {
        final v = getVal(r, k);
        if (v.isNotEmpty) {
          airlines.add(v);
          break;
        }
      }
    }
    totalAirlines = airlines.length;

    // Destinations: COUNT UNIQUE values from DESTINATION_AIRPORT column specifically (case-insensitive)
    final dests = <String>{};
    for (final r in rows) {
      final d = getVal(r, 'DESTINATION_AIRPORT');
      if (d.isNotEmpty) dests.add(d);
    }
    totalDestinations = dests.length;

    // Flight tails (unique)
    final fTails = <String>{};
    for (final r in rows) {
      final t = getFlightTail(r);
      if (t.isNotEmpty) fTails.add(t);
    }
    flightTails = fTails.toList()..sort();

    // Flight routes - use ORIGIN + DESTINATION_AIRPORT where available
    final froutes = <String>{};
    for (final r in rows) {
      final o = getOrigin(r);
      final d = getVal(
        r,
        'DESTINATION_AIRPORT',
      ); // prioritize this column for route destination
      final dest = d.isNotEmpty ? d : getDestination(r);
      if (o.isNotEmpty && dest.isNotEmpty) froutes.add('$o → $dest');
    }
    flightRoutes = froutes.toList()..sort();

    // reset selections if gone
    if (!flightTails.contains(selectedFlightTail)) selectedFlightTail = null;
    if (!flightRoutes.contains(selectedFlightRoute)) selectedFlightRoute = null;
  }

  Map<String, String>? getSelectedRow() {
    if (!csvUploaded) return null;
    if (selectedFlightTail != null) {
      for (final r in rows) {
        if (getFlightTail(r) == selectedFlightTail) return r;
      }
    }
    if (selectedFlightRoute != null) {
      final parts = selectedFlightRoute!
          .split('→')
          .map((s) => s.trim())
          .toList();
      if (parts.length == 2) {
        final o = parts[0], d = parts[1];
        for (final r in rows) {
          final ro = getOrigin(r);
          final rd = getVal(r, 'DESTINATION_AIRPORT').isNotEmpty
              ? getVal(r, 'DESTINATION_AIRPORT')
              : getDestination(r);
          if (ro == o && rd == d) return r;
        }
      }
    }
    return null;
  }

  Widget buildInfoCard(String title, String value) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 6)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget buildCsvPreview() {
    if (!csvUploaded || headers.isEmpty)
      return const Center(child: Text('No CSV loaded'));
    final previewRows = rows.take(20).toList();
    final columns = headers
        .map(
          (h) =>
              DataColumn(label: Text(h, style: const TextStyle(fontSize: 12))),
        )
        .toList();
    final dataRowsWidgets = previewRows.map((r) {
      return DataRow(
        cells: headers.map((h) {
          final val = r[h] ?? '';
          return DataCell(Text(val, style: const TextStyle(fontSize: 12)));
        }).toList(),
      );
    }).toList();
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(columns: columns, rows: dataRowsWidgets),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Runway AI - Overview'),
        backgroundColor: Colors.blue,
      ),
      bottomNavigationBar: BottomNav(
        currentIndex: 0,
        datasetRows: csvUploaded ? rows : null,
        flightRow: getSelectedRow(),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Upload your airport operations CSV to generate real-time predictions',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: pickCsvFile,
                  icon: const Icon(Icons.upload_file),
                  label: const Text('Upload CSV'),
                ),
                const SizedBox(width: 12),
                if (csvUploaded)
                  TextButton(
                    onPressed: () {
                      setState(() {
                        csvUploaded = false;
                        headers = [];
                        rows = [];
                        totalFlights = 0;
                        totalAirlines = 0;
                        totalDestinations = 0;
                        flightTails = [];
                        flightRoutes = [];
                        selectedFlightTail = null;
                        selectedFlightRoute = null;
                      });
                    },
                    child: const Text('Clear CSV'),
                  ),
              ],
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: buildInfoCard(
                    'Flights under monitoring',
                    csvUploaded ? totalFlights.toString() : '--',
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: buildInfoCard(
                    'Airlines involved',
                    csvUploaded ? totalAirlines.toString() : '--',
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: buildInfoCard(
                    'Number of destinations',
                    csvUploaded ? totalDestinations.toString() : '--',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 30),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    decoration: InputDecoration(
                      labelText: 'Flight Tail Number',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    value: selectedFlightTail,
                    items: flightTails
                        .map((f) => DropdownMenuItem(value: f, child: Text(f)))
                        .toList(),
                    onChanged: (v) {
                      setState(() {
                        selectedFlightTail = v;
                        final r = getSelectedRow();
                        if (r != null) {
                          selectedFlightRoute =
                              '${getOrigin(r)} → ${getVal(r, 'DESTINATION_AIRPORT').isNotEmpty ? getVal(r, 'DESTINATION_AIRPORT') : getDestination(r)}';
                        }
                      });
                    },
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    decoration: InputDecoration(
                      labelText: 'Flight Route',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    value: selectedFlightRoute,
                    items: flightRoutes
                        .map((r) => DropdownMenuItem(value: r, child: Text(r)))
                        .toList(),
                    onChanged: (v) {
                      setState(() {
                        selectedFlightRoute = v;
                        if (v != null) {
                          final parts = v
                              .split('→')
                              .map((s) => s.trim())
                              .toList();
                          if (parts.length == 2) {
                            final o = parts[0], d = parts[1];
                            final found = rows.firstWhere((row) {
                              final ro = getOrigin(row);
                              final rd =
                                  getVal(row, 'DESTINATION_AIRPORT').isNotEmpty
                                  ? getVal(row, 'DESTINATION_AIRPORT')
                                  : getDestination(row);
                              return ro == o && rd == d;
                            }, orElse: () => <String, String>{});
                            if (found.isNotEmpty) {
                              final maybeTail = getFlightTail(found);
                              if (maybeTail.isNotEmpty)
                                selectedFlightTail = maybeTail;
                            }
                          }
                        }
                      });
                    },
                  ),
                ),
                const SizedBox(width: 16),
                ElevatedButton(
                  onPressed:
                      (csvUploaded &&
                          (selectedFlightTail != null ||
                              selectedFlightRoute != null))
                      ? () {
                          final selected = getSelectedRow();
                          if (selected == null) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text(
                                  'Selected flight not found in CSV',
                                ),
                              ),
                            );
                            return;
                          }
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => AnomalyPage(
                                datasetRows: rows,
                                flightRow: selected,
                              ),
                            ),
                          );
                        }
                      : null,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      vertical: 16,
                      horizontal: 22,
                    ),
                  ),
                  child: const Text('Detect Anomaly'),
                ),
              ],
            ),
            const SizedBox(height: 30),
            const Text(
              'CSV Preview (first 20 rows)',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [
                  BoxShadow(color: Colors.black12, blurRadius: 6),
                ],
              ),
              child: SizedBox(width: double.infinity, child: buildCsvPreview()),
            ),
          ],
        ),
      ),
    );
  }
}

/* =========================
   Anomaly Page (Updated)
   ========================= */
class AnomalyPage extends StatelessWidget {
  final List<Map<String, String>> datasetRows;
  final Map<String, String> flightRow;

  const AnomalyPage({
    super.key,
    required this.datasetRows,
    required this.flightRow,
  });

  Color riskColor(String level) {
    switch (level) {
      case 'high':
        return Colors.red;
      case 'medium':
        return Colors.orange;
      case 'low':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  // --------------------
  // DELAY COLOR LOGIC
  // --------------------
  Color delayRingColor(int minutes) {
    if (minutes >= 60) return Colors.red;
    if (minutes >= 15) return Colors.orange;
    return Colors.green;
  }

  String computeDelayRisk(Map<String, String> r) {
    final depDelay =
        int.tryParse(getVal(r, 'DEP_DELAY')) ??
        int.tryParse(getVal(r, 'ARR_DELAY')) ??
        0;
    if (depDelay >= 60) return 'high';
    if (depDelay >= 15) return 'medium';
    return 'low';
  }

  String computeWeatherRisk(Map<String, String> r) {
    final w = int.tryParse(getVal(r, 'WEATHER_DELAY')) ?? 0;
    if (w >= 60) return 'high';
    if (w > 0) return 'medium';
    return 'low';
  }

  int computeFuelPercent(Map<String, String> r) {
    final raw = getVal(r, 'FUEL_PERCENT');
    if (raw.isNotEmpty) {
      final p = int.tryParse(raw.replaceAll('%', '').trim());
      if (p != null) return p.clamp(0, 100);
    }
    final depDelay = int.tryParse(getVal(r, 'DEP_DELAY')) ?? 0;
    final simulated = 100 - depDelay;
    return simulated.clamp(5, 100);
  }

  Map<String, int> riskSummary() {
    final weatherRisk = computeWeatherRisk(flightRow);
    final delayRisk = computeDelayRisk(flightRow);
    final fuelPercent = computeFuelPercent(flightRow);
    final fuelRisk = (fuelPercent < 25)
        ? 'high'
        : (fuelPercent < 40 ? 'medium' : 'low');

    int h = 0, m = 0, l = 0;
    for (var r in [weatherRisk, delayRisk, fuelRisk]) {
      if (r == 'high')
        h++;
      else if (r == 'medium')
        m++;
      else
        l++;
    }

    return {'high': h, 'medium': m, 'low': l};
  }

  Widget buildCard({
    required IconData icon,
    required String title,
    required String risk,
    required Widget body,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 6)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 26),
              const SizedBox(width: 10),
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              const Spacer(),
              Container(
                width: 14,
                height: 14,
                decoration: BoxDecoration(
                  color: riskColor(risk),
                  shape: BoxShape.circle,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          body,
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final summary = riskSummary();
    final weatherRisk = computeWeatherRisk(flightRow);
    final delayRisk = computeDelayRisk(flightRow);

    final weatherDelay = int.tryParse(getVal(flightRow, 'WEATHER_DELAY')) ?? 0;
    final depDelay = int.tryParse(getVal(flightRow, 'DEP_DELAY')) ?? 0;
    final securityDelay =
        int.tryParse(getVal(flightRow, 'SECURITY_DELAY')) ?? 0;

    final fuelPercent = computeFuelPercent(flightRow);
    final fuelRisk = (fuelPercent < 25)
        ? 'high'
        : (fuelPercent < 40 ? 'medium' : 'low');

    final predictedDelay = weatherDelay + depDelay + securityDelay;

    // Build times
    final schArrRaw = getVal(flightRow, 'CRS_ARR_TIME');
    final schArr = schArrRaw.padLeft(4, '0');
    final scheduledArrival = "${schArr.substring(0, 2)}:${schArr.substring(2)}";

    final predictedArrivalDT = DateTime(
      2024,
      1,
      1,
      int.parse(scheduledArrival.substring(0, 2)),
      int.parse(scheduledArrival.substring(3)),
    ).add(Duration(minutes: predictedDelay));

    final predictedArrival =
        "${predictedArrivalDT.hour.toString().padLeft(2, '0')}:${predictedArrivalDT.minute.toString().padLeft(2, '0')}";

    final flightId =
        "${getVal(flightRow, 'TAIL_NUMBER')} / ${getVal(flightRow, 'MKT_CARRIER_FL_NUM')}";

    return Scaffold(
      appBar: AppBar(
        title: const Text('Anomaly Detection'),
        backgroundColor: Colors.blue,
      ),
      bottomNavigationBar: BottomNav(
        currentIndex: 1,
        datasetRows: datasetRows,
        flightRow: flightRow,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ---------------------------
            // RISK SUMMARY ROW
            // ---------------------------
            Row(
              children: [
                _riskBadge("High risk", summary['high']!, Colors.red.shade100),
                const SizedBox(width: 12),
                _riskBadge(
                  "Medium risk",
                  summary['medium']!,
                  Colors.orange.shade100,
                ),
                const SizedBox(width: 12),
                _riskBadge("Low risk", summary['low']!, Colors.green.shade100),
              ],
            ),

            const SizedBox(height: 28),

            // ---------------------------
            // DELAY SECTION (large ring)
            // ---------------------------
            Center(
              child: Column(
                children: [
                  const Text(
                    "Predicted Arrival Delay",
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: 140,
                    height: 140,
                    child: CustomPaint(
                      painter: DelayRingPainter(
                        predictedDelay,
                        ringColor: delayRingColor(predictedDelay),
                      ),
                      child: Center(
                        child: Text(
                          "$predictedDelay min",
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    "Expected Arrival: $scheduledArrival\nPredicted Arrival: $predictedArrival",
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 14),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 22),

            // ---------------------------
            // THREE CARDS LAYOUT
            // ---------------------------
            LayoutBuilder(
              builder: (context, constraints) {
                final isNarrow = constraints.maxWidth < 800;

                Widget weatherCard = buildCard(
                  icon: Icons.cloud,
                  title: "Weather Disruption",
                  risk: weatherRisk,
                  body: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Flight Tail Number: $flightId"),
                      const SizedBox(height: 6),
                      Text(
                        "Flight Route: ${getVal(flightRow, 'ORIGIN_AIRPORT')} → ${getVal(flightRow, 'DESTINATION_AIRPORT')}",
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "Forecasted delay: ${weatherDelay >= 60
                            ? '≥ 60 min'
                            : weatherDelay > 0
                            ? '$weatherDelay min'
                            : 'No major forecast'}",
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                );

                Widget carrierCard = buildCard(
                  icon: Icons.airplane_ticket,
                  title: "Carrier Disorder",
                  risk: delayRisk,
                  body: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Flight Tail Number: $flightId"),
                      const SizedBox(height: 6),
                      Text(
                        "Flight Route: ${getVal(flightRow, 'ORIGIN_AIRPORT')} → ${getVal(flightRow, 'DESTINATION_AIRPORT')}",
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "Forecasted delay: ${depDelay >= 60
                            ? '≥ 60 min'
                            : depDelay >= 15
                            ? '$depDelay min'
                            : 'No major forecast'}",
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                );

                Widget securityCard = buildCard(
                  icon: Icons.security,
                  title: "Security Prolongation",
                  risk: securityDelay >= 60
                      ? 'high'
                      : (securityDelay > 0 ? 'medium' : 'low'),
                  body: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Flight Tail Number: $flightId"),
                      const SizedBox(height: 6),
                      Text(
                        "Flight Route: ${getVal(flightRow, 'ORIGIN_AIRPORT')} → ${getVal(flightRow, 'DESTINATION_AIRPORT')}",
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "Delay Forecast: ${securityDelay >= 60
                            ? '≥ 60 min'
                            : securityDelay > 0
                            ? '$securityDelay min'
                            : 'No major forecast'}",
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                );

                if (isNarrow) {
                  return Column(
                    children: [
                      const SizedBox(height: 12),
                      weatherCard,
                      const SizedBox(height: 18),
                      carrierCard,
                      const SizedBox(height: 18),
                      securityCard,
                    ],
                  );
                } else {
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(child: weatherCard),
                      const SizedBox(width: 14),
                      Expanded(child: carrierCard),
                      const SizedBox(width: 14),
                      Expanded(child: securityCard),
                    ],
                  );
                }
              },
            ),

            const SizedBox(height: 26),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pushNamed(context, '/optimize'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
                child: const Text(
                  'View Optimization',
                  style: TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // badge widget helper
  Widget _riskBadge(String label, int count, Color bg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Text(label),
          const SizedBox(width: 8),
          CircleAvatar(
            radius: 12,
            backgroundColor: Colors.white,
            child: Text(count.toString()),
          ),
        ],
      ),
    );
  }
}

/* =========================
   Optimization Page (placeholder)
   ========================= */
class OptimizationPage extends StatelessWidget {
  const OptimizationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Optimization'),
        backgroundColor: Colors.blue,
      ),
      bottomNavigationBar: const BottomNav(currentIndex: 2),
      body: const Center(
        child: Text(
          'Optimization suggestions will be shown here.',
          style: TextStyle(fontSize: 18),
        ),
      ),
    );
  }
}

class DelayRingPainter extends CustomPainter {
  final int minutes;
  final Color ringColor;
  final double strokeWidth;

  DelayRingPainter(
    this.minutes, {
    required this.ringColor,
    this.strokeWidth = 12.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width, size.height) / 2 - strokeWidth;

    // background ring
    final bgPaint = Paint()
      ..color = Colors.grey.shade200
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;

    // progress ring
    final progressPaint = Paint()
      ..color = ringColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawCircle(center, radius, bgPaint);

    // percentage relative to a chosen cap (120 minutes)
    final pct = (minutes / 120).clamp(0.0, 1.0);
    final sweep = 2 * pi * pct;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -pi / 2, // start at top
      sweep,
      false,
      progressPaint,
    );
  }

  @override
  bool shouldRepaint(covariant DelayRingPainter oldDelegate) {
    return oldDelegate.minutes != minutes ||
        oldDelegate.ringColor != ringColor ||
        oldDelegate.strokeWidth != strokeWidth;
  }
}
